"""Tool surface exposed to the reasoning agent.

The agent never touches Python objects directly: it emits a JSON tool call, the
runtime executes it against the *real* backend, and a compact JSON result goes
back into the conversation. Every tool is bounded (see `editing.*` in config)
and read-only except the ones that mutate: apply_edit, revert,
revert_transaction.

Revert semantics (V4)
    ``revert(n)``             undo the newest n EDITS, newest-first.
    ``revert_transaction(id)`` undo an ENTIRE transaction, including a
                              transaction that was already committed.
    ``cascade=True``          allow the dependency-aware cascade (revert the
                              newer transactions that share a tensor first).

Invariants enforced here:
* an edit ID is reported as undone ONLY after a hash-verified tensor restore;
* a failed restore changes no tracking and no ledger status;
* every edit's status is written to the transaction that OWNS it (`txn_id` of
  the edit record), never to "the latest transaction";
* reverting an older transaction while a newer one still holds a value of the
  same tensor is refused (or cascaded newest-first) instead of corrupting it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from ..edit_ledger import (INACTIVE_EDIT_STATUSES, INACTIVE_TXN_STATUSES,
                           EditLedger, EditRecord, EditTransaction, LedgerError,
                           RestoreError)
from ..evaluation.runner import compare, run_eval
from ..inspection.activations import group_contrast
from ..inspection.tensors import diff_tensors, suggest_editable, tensor_stats
from ..tensor_editor.engine import EditRefused


class RevertConflict(RuntimeError):
    """A requested revert would leave tensors/transactions inconsistent."""


class ToolRuntime:
    def __init__(self, backend, cfg: dict, editor, fwd, dataset, workdir,
                 baseline_result: dict | None = None, task: str = ""):
        self.backend = backend
        self.cfg = cfg
        self.editor = editor
        self.fwd = fwd
        self.dataset = dataset
        self.workdir = workdir
        self.baseline = baseline_result
        self.task = task
        self.history: List[dict] = []
        self.applied: List[dict] = []
        self.candidates: List[dict] = []
        self.last_candidate: dict | None = None
        # multi-edit transaction state (single source of truth for one interaction)
        self.ledger = EditLedger(Path(workdir) / "edit_ledger")
        self.txn: EditTransaction | None = None
        self.last_txn_id: str | None = None      # informational only; never an owner
        # edit IDs already undone (verified restore); makes repeated reverts no-ops
        self._reverted_ids: set = set()
        # tensor -> {hash -> pristine float32 pre-image} across ALL transactions of
        # this run, keyed by the hash of the tensor state IMMEDIATELY BEFORE the
        # edit that needs undoing, so one edit out of several on the same tensor
        # can be restored exactly.
        self._pristine_preimages: Dict[str, Dict[str, np.ndarray]] = {}
        # strict execution order of every applied edit (also mirrors rt.applied)
        self._edit_order: List[str] = []
        # records recovered from the durable ledger (survives a restart)
        self._records_cache: Dict[str, EditRecord] = {}
        self.recovery = self._restore_from_ledger()

    # ------------------------------------------------------- restart recovery
    def _restore_from_ledger(self) -> dict:
        """Rebuild active-edit tracking from the durable ledger.

        Only transactions whose outcome is still live contribute. For the first
        edit on a tensor inside a transaction the on-disk journal is the exact
        pre-image, so a per-edit revert can work after a restart; later edits on
        the same tensor remain revertible through their whole transaction.
        """
        report = {"recovered_transactions": [], "recovered_edits": 0,
                  "journal_preimages": 0, "skipped_transactions": []}
        for t in self.ledger.transactions():
            tid = t["txn_id"]
            if t.get("status") in INACTIVE_TXN_STATUSES:
                continue
            try:
                txn = self.ledger.transaction(tid)
            except (LedgerError, json.JSONDecodeError, OSError):
                report["skipped_transactions"].append(tid)
                continue
            reverted = set(t.get("reverted_edit_ids") or [])
            journaled = set()
            recovered = 0
            for e in txn.edits:
                eid = e["edit_id"]
                if eid in reverted or e.get("status") in INACTIVE_EDIT_STATUSES:
                    continue
                rec = EditRecord(dict(e))
                rec["txn_seq"] = t.get("seq", 0)
                self._records_cache[eid] = rec
                self.applied.append({
                    "edit_id": eid, "tensor": e["tensor"], "txn_id": tid,
                    "base_sha256": e["tensor_sha256_before"],
                    "edited_sha256": e["tensor_sha256_after"],
                    "changed_entries": e.get("changed_entries"),
                    "relative_change": e.get("relative_change"),
                    "recovered": True,
                })
                self._edit_order.append(eid)
                recovered += 1
                name = e["tensor"]
                if name not in journaled:
                    p = txn._journal_path(name)
                    if p.exists():
                        slot = self._pristine_preimages.setdefault(name, {})
                        if e["tensor_sha256_before"] not in slot:
                            slot[e["tensor_sha256_before"]] = np.load(p)
                            journaled.add(name)
                            report["journal_preimages"] += 1
            if recovered:
                report["recovered_transactions"].append(tid)
                report["recovered_edits"] += recovered
        return report

    # --------------------------------------------------------- transactions
    def begin_transaction(self) -> EditTransaction:
        """Open a fresh edit transaction for one candidate interaction."""
        self.commit_transaction()
        self.txn = EditTransaction(self.backend, self.ledger.root)
        return self.txn

    def capture_preimage(self, name: str) -> None:
        """Snapshot the tensor's CURRENT bits, keyed by their current hash."""
        h = self.backend.tensor_hash(name)
        slot = self._pristine_preimages.setdefault(name, {})
        if h not in slot:
            slot[h] = self.backend.get_tensor(name)

    def _own_edit_ids(self, txn: EditTransaction) -> List[str]:
        return txn.edit_ids()

    def commit_transaction(self) -> dict:
        """Close the open transaction as committed (kept edits await accept/reject)."""
        if self.txn is None or self.txn.status != "open":
            return {"txn_id": self.txn.txn_id if self.txn else None, "committed": False}
        for name, h in self.txn._saved.items():
            self._pristine_preimages.setdefault(name, {})[h] = self.txn._preimage(name)
        self.txn.mark("committed")
        if self.txn.edits:                          # never register empty txns
            self.ledger.register(self.txn, "committed", None)
        self.last_txn_id = self.txn.txn_id          # informational, not ownership
        out = {"txn_id": self.txn.txn_id, "committed": True,
               "edit_ids": self.txn.edit_ids(), "tensors": self.txn.tensors_touched()}
        for e in self.txn.edits:
            self._records_cache[e["edit_id"]] = EditRecord(dict(e))
        self.txn = None
        return out

    def rollback_transaction(self, error: str | None = None,
                             status: str | None = None) -> dict:
        """Atomically undo every edit of the OPEN transaction and register it.

        status: 'rejected' (candidate turned down) or 'failed' (an exception
        interrupted the transaction). A restore failure raises RestoreError after
        the ledger has been left untouched and the tensors have been recovered.
        """
        if self.txn is None:
            return {"ok": True, "note": "no open transaction", "restored": [],
                    "undone": [], "remaining": [a.get("edit_id") for a in self.applied]}
        txn = self.txn
        try:
            rep = txn.restore_pristine(verify=True)
        except RestoreError as ex:
            txn.mark("failed", error=f"{error}; restore error: {ex}")
            self.ledger.register(txn, "failed", False, str(ex))
            self.txn = None
            raise
        final_status = status or ("failed" if error else "rejected")
        undone = [e for e in txn.edit_ids() if e not in self._reverted_ids]
        txn.mark(final_status, error=error)
        self.ledger.register(txn, final_status, False, error)
        if rep["ok"]:
            self._reverted_ids.update(undone)
            self.applied = [a for a in self.applied
                            if a.get("edit_id") not in set(undone)]
            self._edit_order = [e for e in self._edit_order if e not in set(undone)]
            for eid in undone:
                self._records_cache.pop(eid, None)
        out = {"ok": rep["ok"], "txn_id": rep["txn_id"], "status": final_status,
               "error": error, "undone": undone,
               "remaining": [a.get("edit_id") for a in self.applied],
               "atomic": True, **rep}
        self.txn = None
        return out

    # ------------------------------------------------------------ read-only
    def t_list_tensors(self, pattern: str = "", limit: int = 40) -> dict:
        return {"tensors": suggest_editable(self.backend, self.cfg, prefer=pattern or "down_proj")[:limit]}

    def t_tensor_stats(self, name: str) -> dict:
        if name not in set(self.backend.param_names()):
            return {"error": f"unknown tensor {name!r}"}
        return tensor_stats(self.backend, name)

    def t_forward_generate(self, prompts: List[str], max_new_tokens: int = 32) -> dict:
        rec = self.fwd.generate(prompts, max_new_tokens=max_new_tokens,
                                temperature=0.0, seed=0)
        return {"id": rec["id"], "outputs": rec["outputs"],
                "budget_left": self.fwd.remaining()}

    def t_activation_contrast(self, hindi_prompts: List[str], english_prompts: List[str]) -> dict:
        c = group_contrast(self.backend, hindi_prompts, english_prompts, top_k=10)
        return {"ranked_tensors": c["ranked"],
                "per_tensor": {k: {kk: vv for kk, vv in v.items() if kk != "top_channels"}
                               for k, v in list(c["per_tensor"].items())[:6]}}

    def t_preview_edit(self, operation: dict) -> dict:
        p = self.editor.preview(operation)
        if "diff" in p:
            p["diff"].pop("examples", None)
        return p

    def t_eval_subset(self, limit_per_category: int = 4) -> dict:
        res = run_eval(self.backend, self.dataset, self.cfg, out_dir=None,
                       label="probe", limit_per_category=limit_per_category)
        return {"composite": res["aggregate"]["composite"],
                "categories": {k: v["mean"] for k, v in res["aggregate"]["categories"].items()},
                "n": res["aggregate"]["n"]}

    def t_ledger_status(self) -> dict:
        """Live view: which edits are active, which transactions own them."""
        return {"active_edit_ids": [a.get("edit_id") for a in self.applied],
                "reverted_edit_ids": sorted(self._reverted_ids),
                "open_transaction": self.txn.txn_id if self.txn else None,
                "ledger": self.ledger.status_report(),
                "recovery": self.recovery}

    # --------------------------------------------------------------- mutating
    def t_apply_edit(self, operation: dict, reason: str = "", hypothesis: str = "") -> dict:
        if self.txn is None:
            self.begin_transaction()
        try:
            name = operation.get("tensor") or operation.get("name")
            self.capture_preimage(name)             # per-edit pre-image (hash-keyed)
            self.txn.capture_pristine(name)          # pre-transaction bits, once per tensor
            out = self.editor.apply(operation, reason=reason, meta={
                "hypothesis": hypothesis, "experiment": self.workdir.name,
                "txn_id": self.txn.txn_id,
                "agent": self.cfg["agent"].get("model") or "scripted"})
        except EditRefused as e:
            return {"ok": False, "refused": str(e)}
        except Exception as e:
            # a failure mid-transaction must never leave partially edited weights
            rep = self.rollback_transaction(error=str(e))
            return {"ok": False, "error": str(e), "transaction_rollback": rep}
        out.pop("manifest", None)
        rec = self.txn.record_edit(out["edit_id"], out["tensor"], {
            "edited_sha256": out["edited_sha256"], "base_sha256": out["base_sha256"],
            "shape": [int(x) for x in self.backend.shape_of(out["tensor"])],
            "dtype": self.backend.dtype_of(out["tensor"]),
            "changed_entries": out["changed_entries"],
            "relative_change": out["relative_change"],
        }, {"reason": reason, "hypothesis": hypothesis})
        self._records_cache[rec["edit_id"]] = rec
        out["txn_id"] = self.txn.txn_id
        self.applied.append(out)
        self._edit_order.append(out["edit_id"])
        return {"ok": True, **out, "n_edits_in_transaction": len(self.txn.edits)}

    def t_evaluate_candidate(self, limit_per_category: int = 6) -> dict:
        cand = run_eval(self.backend, self.dataset, self.cfg, out_dir=None,
                        label="candidate", limit_per_category=limit_per_category)
        decision = compare(self.baseline, cand, self.cfg) if self.baseline else {
            "accept": None, "reason": "no baseline recorded"}
        self.last_candidate = cand
        self.candidates.append({"result": cand, "decision": decision})
        return {"composite": cand["aggregate"]["composite"],
                "delta": decision.get("delta"),
                "accept": decision.get("accept"),
                "reason": decision.get("reason"),
                "regressions": decision.get("regressions"),
                "categories": {k: v["mean"] for k, v in cand["aggregate"]["categories"].items()}}

    # ------------------------------------------------------------- revert V4
    def _records(self) -> Dict[str, EditRecord]:
        """All edit records of this run, in execution order (open txn included)."""
        recs: Dict[str, EditRecord] = dict(self._records_cache)
        if self.txn is not None and self.txn.edits:
            for e in self.txn.edits:
                recs[e["edit_id"]] = e
        for i, a in enumerate(self.applied):
            eid = a["edit_id"]
            if eid in recs:
                continue
            recs[eid] = EditRecord({
                "edit_id": eid, "txn_id": a.get("txn_id"),
                "tensor": a["tensor"],
                "tensor_sha256_after": a.get("edited_sha256"),
                "tensor_sha256_before": a.get("base_sha256"),
                "txn_order": i,
            })
        return recs

    def _order_index(self) -> Dict[str, int]:
        return {eid: i for i, eid in enumerate(self._edit_order)}

    def _revert_one_edit(self, rec: EditRecord, verify: bool = True) -> str:
        """Restore ONE tensor to the pre-image that existed before this edit.

        Returns the edit_id on success; raises with the original error detail on
        any failure (tracking is not touched here).
        """
        name = rec["tensor"]
        before = rec["tensor_sha256_before"]
        pre = self._pristine_preimages.get(name, {}).get(before)
        if pre is None and (self.txn is not None
                            and rec.get("txn_id") == self.txn.txn_id
                            and name in self.txn._saved
                            and self.txn._saved[name] == before):
            pre = self.txn._preimage(name)
        if pre is None:
            raise RuntimeError(
                f"no pristine pre-image saved for tensor {name!r} "
                f"(base hash {(before or '')[:12]}); this edit cannot be "
                "individually reverted - use revert_transaction(txn_id)")
        from ..sparse_deltas.cast import cast_to_dtype
        from ..sparse_deltas.store import sha256_arr
        if verify:
            h = sha256_arr(cast_to_dtype(pre, self.backend.dtype_of(name)))
            if h != before:
                raise RuntimeError(
                    f"{name}: pristine pre-image hash mismatch ({h[:12]} != "
                    f"{before[:12]}); tensor NOT modified")
        self.backend.set_tensor(name, pre)
        if verify and self.backend.tensor_hash(name) != before:
            raise RuntimeError(f"{name}: post-restore hash mismatch; rollback NOT verified")
        return rec["edit_id"]

    def _validate_partial_revert(self, target_recs: List[EditRecord]) -> None:
        """Refuse an out-of-order revert that would break a same-tensor dependency."""
        recs = self._records()
        order = self._order_index()
        target_ids = {r["edit_id"] for r in target_recs}
        for rec in target_recs:
            mine = order.get(rec["edit_id"], rec.get("txn_order", 0))
            for other in recs.values():
                if (other.get("tensor") == rec.get("tensor")
                        and other["edit_id"] != rec["edit_id"]
                        and other["edit_id"] not in target_ids
                        and order.get(other["edit_id"], other.get("txn_order", 0)) > mine
                        and other["edit_id"] not in self._reverted_ids):
                    raise RevertConflict(
                        f"refusing to revert {rec['edit_id']}: newer edit "
                        f"{other['edit_id']} on the same tensor {rec['tensor']} is "
                        "still applied; revert it first (newest-first)")

    def _group_by_owner(self, edit_ids: List[str], recs: Dict[str, EditRecord]) -> Dict[str, List[str]]:
        """Map owning transaction -> its edit IDs. NEVER uses `last_txn_id`."""
        out: Dict[str, List[str]] = {}
        for eid in edit_ids:
            owner = (recs.get(eid) or {}).get("txn_id")
            if not owner:
                owner = self.ledger.owner_of(eid)
            if owner:
                out.setdefault(owner, []).append(eid)
        return out

    def _apply_revert_bookkeeping(self, undone: List[str]) -> dict:
        """Update every tracking surface after verified restores.

        Ledger status is written per OWNING transaction, so edits reverted from
        transactions A and B update A and B separately.
        """
        recs = self._records()
        by_txn = self._group_by_owner(undone, recs)
        updated = []
        if self.txn is not None:
            mine = by_txn.pop(self.txn.txn_id, [])
            if mine:
                self.txn.mark_edits_reverted(mine, None)
        for tid, ids in by_txn.items():
            updated.append(self.ledger.mark_edits_reverted(tid, ids, None))
        self._reverted_ids.update(undone)
        undone_set = set(undone)
        self.applied = [a for a in self.applied if a.get("edit_id") not in undone_set]
        self._edit_order = [e for e in self._edit_order if e not in undone_set]
        for eid in undone_set:
            self._records_cache.pop(eid, None)
        return {"ledger_updates": updated, "by_transaction": by_txn}

    def t_revert(self, n: int = 1) -> dict:
        """Undo the last n APPLIED EDITS, newest first (n = number of edits).

        Each edit's tensor is restored to the exact pristine bits saved before
        that edit, and the SHA-256 is verified AFTER the write. Only then are the
        edit IDs reported as "undone" and removed from applied tracking. Use
        revert_transaction(txn_id) to undo an entire (including committed)
        transaction.
        """
        n = max(1, int(n))
        recs = self._records()
        live = [rid for rid in self._edit_order
                if rid in recs and rid not in self._reverted_ids]
        if not live:
            return {"ok": True, "undone": [], "restored": [], "remaining":
                    [a.get("edit_id") for a in self.applied], "verified": True,
                    "txn_id": self.txn.txn_id if self.txn else None, "error": None,
                    "note": "nothing applied; idempotent no-op"}
        if n > len(live):
            return {"ok": False,
                    "error": f"requested n={n} but only {len(live)} live edit(s) "
                             "applied; use revert_transaction() to undo a whole "
                             "transaction, or a smaller n",
                    "undone": [], "restored": [], "remaining": live,
                    "verified": False, "txn_id": None}
        target_ids = list(reversed(live))[:n]              # newest first
        recs_by_id = {r["edit_id"]: r for r in recs.values()}
        target_recs = [recs_by_id[i] for i in target_ids]
        try:
            self._validate_partial_revert(target_recs)
            undone, restored = [], []
            for rec in target_recs:
                undone.append(self._revert_one_edit(rec))
                restored.append(rec["tensor"])
        except RevertConflict as e:
            return {"ok": False, "error": str(e), "undone": [], "restored": [],
                    "remaining": live, "verified": False,
                    "txn_id": self.txn.txn_id if self.txn else None}
        except Exception as ex:
            # FAILED (possibly mid-way): only the verified restores are undone.
            done = list(undone)
            if done:
                self._apply_revert_bookkeeping(done)
            return {"ok": False, "error": f"{type(ex).__name__}: {ex}",
                    "undone": done, "restored": restored,
                    "remaining": [a.get("edit_id") for a in self.applied],
                    "verified": False,
                    "txn_id": self.txn.txn_id if self.txn else self.last_txn_id}
        newest_owner = (recs.get(undone[0]) or {}).get("txn_id") if undone else None
        book = self._apply_revert_bookkeeping(undone)
        return {"ok": True, "undone": undone, "restored": restored,
                "remaining": [a.get("edit_id") for a in self.applied],
                "verified": True,
                "txn_id": newest_owner or (self.txn.txn_id if self.txn else None),
                "error": None, "ledger_updates": book["ledger_updates"]}

    def revert_transaction(self, txn_id: str | None = None,
                           cascade: bool = False) -> dict:
        """Atomically undo an ENTIRE transaction, including a committed one.

        Design (documented in docs/REVERT.md):
        * `txn_id=None` targets the open transaction.
        * A committed transaction is revertible as long as its pristine journal
          still exists. The restore is pre-validated and hash-verified; the
          ledger and tracking change only after the restore succeeded.
        * If a LATER transaction still has edits on one of the same tensors, a
          standalone revert is refused with `blocked_by` and nothing is modified.
          `cascade=True` reverts those dependents first, newest-first.
        * Repeating a revert is an idempotent no-op (`undone: []`).
        * Unknown transaction IDs return a clear error.
        """
        open_id = self.txn.txn_id if self.txn is not None else None
        if self.txn is not None and (txn_id is None or txn_id == open_id):
            rep = self.rollback_transaction(status="rejected",
                                            error="explicit revert_transaction")
            return {"ok": rep["ok"], "undone": rep["undone"],
                    "restored": rep["restored"], "remaining": rep["remaining"],
                    "verified": bool(rep.get("ok")), "txn_id": rep["txn_id"],
                    "error": rep.get("error"), "atomic": True,
                    "note": "open transaction reverted"}
        if txn_id is None:
            return {"ok": True, "undone": [], "restored": [], "remaining":
                    [a.get("edit_id") for a in self.applied], "verified": True,
                    "txn_id": None, "error": None,
                    "note": "no open transaction; idempotent no-op"}

        if not self.ledger.has(txn_id):
            return {"ok": False, "undone": [], "restored": [],
                    "remaining": [a.get("edit_id") for a in self.applied],
                    "verified": False, "txn_id": txn_id,
                    "error": f"unknown transaction {txn_id!r}; known transactions: "
                             f"{[t['txn_id'] for t in self.ledger.transactions()]}"}
        try:
            txn = self.ledger.transaction(txn_id, backend=self.backend)
        except (LedgerError, json.JSONDecodeError, OSError) as e:
            return {"ok": False, "undone": [], "restored": [],
                    "remaining": [a.get("edit_id") for a in self.applied],
                    "verified": False, "txn_id": txn_id,
                    "error": f"cannot load transaction {txn_id!r}: {e}"}

        idx = self.ledger.get(txn_id) or {}
        already = set(idx.get("reverted_edit_ids") or []) | self._reverted_ids
        live = [e["edit_id"] for e in txn.edits
                if e["edit_id"] not in already
                and e.get("status") not in INACTIVE_EDIT_STATUSES]
        if not live:
            return {"ok": True, "undone": [], "restored": [],
                    "remaining": [a.get("edit_id") for a in self.applied],
                    "verified": True, "txn_id": txn_id, "error": None,
                    "note": "transaction already reverted; idempotent no-op"}

        # ---- dependency check BEFORE any write ---------------------------
        deps = self.ledger.dependents(txn_id)
        if self.txn is not None:
            open_tensors = {e["tensor"] for e in self.txn.edits}
            shared = sorted(set(txn.tensors_touched()) & open_tensors)
            if shared:
                deps.append({"txn_id": self.txn.txn_id,
                             "edit_ids": self.txn.edit_ids(),
                             "tensors": shared, "status": "open"})
        if deps and not cascade:
            return {"ok": False, "undone": [], "restored": [],
                    "remaining": [a.get("edit_id") for a in self.applied],
                    "verified": False, "txn_id": txn_id, "blocked_by": deps,
                    "error": ("refusing to revert transaction "
                              f"{txn_id!r} in isolation: later transaction(s) "
                              f"{[d['txn_id'] for d in deps]} still have edits on "
                              "the same tensor(s) "
                              f"{sorted({t for d in deps for t in d['tensors']})}. "
                              "Revert those first (newest-first) or pass "
                              "cascade=True."),
                    "state_changed": False}

        cascaded = []
        if deps and cascade:
            for dep in reversed(deps):                  # newest first
                if self.txn is not None and dep["txn_id"] == self.txn.txn_id:
                    r = self.rollback_transaction(
                        error=f"cascade for revert_transaction({txn_id})",
                        status="rejected")
                    r = {"ok": r.get("ok"), "undone": r.get("undone", []),
                         "error": r.get("error")}
                else:
                    r = self.revert_transaction(dep["txn_id"], cascade=True)
                if not r.get("ok"):
                    return {"ok": False, "undone": [], "restored": [],
                            "remaining": [a.get("edit_id") for a in self.applied],
                            "verified": False, "txn_id": txn_id,
                            "cascaded": cascaded,
                            "error": (f"cascade failed while reverting "
                                      f"{dep['txn_id']!r}: {r.get('error')}")}
                cascaded.append(dep["txn_id"])

        # ---- journal pre-validation: no writes if anything is missing -----
        report = txn.journal_report()
        if report["journal_missing"]:
            msg = ("transaction journal is incomplete for tensors "
                   f"{report['journal_missing']}; refusing to guess the previous "
                   "state. Use the checkpoint chain "
                   "(CheckpointManager.rebuild_chain_after_reset).")
            if cascaded:
                msg = (f"cascade already reverted {cascaded}, but {txn_id!r} has no "
                       f"complete journal: {report['journal_missing']}")
            return {"ok": False, "undone": [], "restored": [],
                    "remaining": [a.get("edit_id") for a in self.applied],
                    "verified": False, "txn_id": txn_id, "blocked_by": deps,
                    "error": msg, "state_changed": False, "cascaded": cascaded,
                    "journal": report}

        # ---- atomic restore ---------------------------------------------
        try:
            rep = txn.restore_pristine(verify=True)
        except RestoreError as ex:
            return {"ok": False, "undone": [], "restored": ex.restored,
                    "remaining": [a.get("edit_id") for a in self.applied],
                    "verified": False, "txn_id": txn_id,
                    "atomic": True, "recovered": ex.recovered,
                    "state_changed": False, "cascaded": cascaded,
                    "error": f"restore failed, transaction NOT marked reverted: {ex}"}
        if not rep.get("ok"):
            return {"ok": False, "undone": [], "restored": rep.get("restored", []),
                    "remaining": [a.get("edit_id") for a in self.applied],
                    "verified": False, "txn_id": txn_id, "atomic": True,
                    "state_changed": False, "cascaded": cascaded,
                    "drift": rep.get("drift"),
                    "error": f"restore produced drift on {rep.get('drift')}; ledger unchanged"}

        # ---- only now: ledger + tracking ---------------------------------
        upd = self.ledger.mark_edits_reverted(txn_id, live, None)
        txn.mark_edits_reverted(live, None)
        self._reverted_ids.update(live)
        live_set = set(live)
        self.applied = [a for a in self.applied if a.get("edit_id") not in live_set]
        self._edit_order = [e for e in self._edit_order if e not in live_set]
        for eid in live_set:
            self._records_cache.pop(eid, None)
        return {"ok": True, "undone": live, "restored": rep["restored"],
                "remaining": [a.get("edit_id") for a in self.applied],
                "verified": True, "txn_id": txn_id, "error": None,
                "atomic": True, "journal": report, "ledger": upd,
                "cascaded": cascaded, "state_changed": True}

    def revert_last_n(self, n: int = 1) -> dict:
        """Loop-facing revert that always restores tensors."""
        return self.t_revert(n=n)

    def t_revert_transaction(self, txn_id: str | None = None,
                             cascade: bool = False) -> dict:
        """Tool-surface entry point for revert_transaction."""
        return self.revert_transaction(txn_id, cascade=cascade)

    def revert_whole_transaction(self, txn_id: str | None = None) -> dict:
        """Loop/agent-facing whole-transaction revert."""
        return self.revert_transaction(txn_id)

    def register_loop_edit(self, applied_now: dict, hypothesis: str = "",
                           agent: str = "unknown") -> EditRecord:
        """Record a loop-applied edit in the open transaction AND applied tracking."""
        if self.txn is None:
            self.begin_transaction()
            self.txn.capture_pristine(applied_now["tensor"])
        rec = self.txn.record_edit(applied_now["edit_id"], applied_now["tensor"], {
            "edited_sha256": applied_now["edited_sha256"],
            "base_sha256": applied_now["base_sha256"],
            "shape": [int(x) for x in self.backend.shape_of(applied_now["tensor"])],
            "dtype": self.backend.dtype_of(applied_now["tensor"]),
            "changed_entries": applied_now["changed_entries"],
            "relative_change": applied_now["relative_change"],
        }, {"hypothesis": hypothesis, "agent": agent})
        self._records_cache[rec["edit_id"]] = rec
        self.applied.append(applied_now)
        self._edit_order.append(applied_now["edit_id"])
        return rec

    def t_finish(self, summary: str) -> dict:
        return {"ack": True, "summary": summary}

    # ---------------------------------------------------------------- schema
    def schema(self) -> List[dict]:
        return [
            {"name": "list_tensors", "args": {"pattern": "str (optional substring filter)"},
             "doc": "List 2-D editable tensors ranked by suitability."},
            {"name": "tensor_stats", "args": {"name": "str"},
             "doc": "Norms, sparsity, top columns/rows, rank for one tensor."},
            {"name": "forward_generate", "args": {"prompts": "[str]", "max_new_tokens": "int"},
             "doc": "Run a budgeted generation pass on the current weights."},
            {"name": "activation_contrast",
             "args": {"hindi_prompts": "[str]", "english_prompts": "[str]"},
             "doc": "Which layers/channels fire more on Hindi than English."},
            {"name": "preview_edit", "args": {"operation": "op object"},
             "doc": "Validate + measure a candidate edit WITHOUT applying it."},
            {"name": "apply_edit", "args": {"operation": "op object", "reason": "str",
                                            "hypothesis": "str"},
             "doc": "Apply a bounded direct tensor edit (hash-verified)."},
            {"name": "eval_subset", "args": {"limit_per_category": "int"},
             "doc": "Quick dev-split score, no accept/reject."},
            {"name": "evaluate_candidate", "args": {"limit_per_category": "int"},
             "doc": "Full dev evaluation + accept/reject against baseline."},
            {"name": "revert", "args": {"n": "int"},
             "doc": "Undo the last n APPLIED EDITS (n counts edits, newest first; "
                    "exact verified tensor restore)."},
            {"name": "revert_transaction",
             "args": {"txn_id": "str (optional; committed IDs are allowed)",
                      "cascade": "bool (revert newer dependent transactions first)"},
             "doc": "Atomically undo an ENTIRE transaction, committed or open."},
            {"name": "ledger_status", "args": {},
             "doc": "Which edits are active/reverted and which transaction owns them."},
            {"name": "finish", "args": {"summary": "str"},
             "doc": "End the experiment with a summary."},
        ]

    def call(self, name: str, args: Dict) -> dict:
        fn: Callable | None = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name!r}",
                    "available": [s["name"] for s in self.schema()]}
        try:
            out = fn(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:                       # tool errors must not kill the loop
            return {"error": f"{type(e).__name__}: {e}"}
        self.history.append({"tool": name, "args": _small(args), "result": _small(out)})
        return out


def _small(o, depth: int = 0):
    """Keep tool results small enough for a chat context."""
    if depth > 3:
        return "<...>"
    if isinstance(o, dict):
        return {k: _small(v, depth + 1) for k, v in list(o.items())[:24]}
    if isinstance(o, (list, tuple)):
        return [_small(v, depth + 1) for v in list(o)[:12]]
    if isinstance(o, float):
        return round(o, 5)
    if isinstance(o, (np.floating,)):
        return round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, str) and len(o) > 400:
        return o[:400] + "..."
    return o


def parse_tool_call(text: str) -> dict | None:
    """Extract one JSON tool call from a model reply.

    Accepts ```json fenced blocks, bare JSON objects, and the common
    {"tool": ..., "args": ...} / {"name": ..., "arguments": ...} shapes.
    """
    if not text:
        return None
    import re
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)
    cands = blocks + [text]
    for blk in cands:
        blk = blk.strip()
        for m in re.finditer(r"\{", blk):
            frag = blk[m.start():]
            depth, end = 0, None
            for i, ch in enumerate(frag):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end is None:
                continue
            try:
                obj = json.loads(frag[:end])
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            name = obj.get("tool") or obj.get("name") or obj.get("function")
            args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name:
                return {"tool": name, "args": args, "raw": obj}
    return None


def parse_operations(tool_input: str) -> dict | None:
    """Parse the final submission form: {"operations": [...], "reason": "..."}."""
    return parse_tool_call(tool_input)
