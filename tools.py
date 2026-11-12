"""Tool surface exposed to the reasoning agent.

The agent never touches Python objects directly: it emits a JSON tool call, the
runtime executes it against the *real* backend, and a compact JSON result goes
back into the conversation. Every tool is bounded (see editing.* in config) and
read-only except the three that mutate: apply_edit, evaluate_candidate, revert.

This is the bridge the task asks for: the NVIDIA model reasons, the Python
environment measures and edits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from ..edit_ledger import EditLedger, EditRecord, EditTransaction
from ..evaluation.runner import compare, run_eval
from ..inspection.activations import group_contrast
from ..inspection.tensors import diff_tensors, suggest_editable, tensor_stats
from ..tensor_editor.engine import EditRefused


class RevertConflict(RuntimeError):
    """A requested partial revert would leave a transaction inconsistent."""


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
        self.last_txn_id: str | None = None
        # edit IDs already undone (verified restore); makes repeated reverts no-ops
        self._reverted_ids: set = set()
        # tensor -> {hash -> pristine float32 pre-image} across ALL transactions of
        # this run. Keyed by the hash of the tensor state IMMEDIATELY BEFORE the
        # edit that needs undoing (captured per edit), so a single edit out of
        # several on the same tensor can be restored exactly. The per-transaction
        # journal alone only stores the pre-TRANSACTION state, which cannot undo
        # one edit out of several on the same tensor.
        self._pristine_preimages: Dict[str, Dict[str, np.ndarray]] = {}
        # strict execution order of every applied edit (also mirrors rt.applied)
        self._edit_order: List[str] = []

    # --------------------------------------------------------- transactions
    def begin_transaction(self) -> EditTransaction:
        """Open a fresh edit transaction for one candidate interaction."""
        self.commit_transaction()
        self.txn = EditTransaction(self.backend, self.ledger.root)
        return self.txn

    def capture_preimage(self, name: str) -> None:
        """Snapshot the tensor's CURRENT bits, keyed by their current hash.

        Called before EVERY edit (not once per transaction): edit k on tensor T
        needs the state just before k, which for k>1 differs from the
        pre-transaction journal entry. Storage cost: one float32 copy per edit.
        """
        h = self.backend.tensor_hash(name)
        slot = self._pristine_preimages.setdefault(name, {})
        if h not in slot:
            slot[h] = self.backend.get_tensor(name)

    def commit_transaction(self) -> dict:
        """Close the open transaction as committed (kept edits await accept/reject)."""
        if self.txn is None or self.txn.status != "open":
            return {"txn_id": self.txn.txn_id if self.txn else None, "committed": False}
        # keep per-tensor pristine pre-images available for LATER per-edit reverts
        # (the per-transaction journal is deleted with the txn dir otherwise)
        for name, h in self.txn._saved.items():
            self._pristine_preimages.setdefault(name, {})[h] = self.txn._preimage(name)
        self.txn.mark("committed")
        self.ledger.register(self.txn, "committed", None)
        self.last_txn_id = self.txn.txn_id
        out = {"txn_id": self.txn.txn_id, "committed": True,
               "edit_ids": self.txn.edit_ids(), "tensors": self.txn.tensors_touched()}
        self.txn = None
        return out

    def rollback_transaction(self, error: str | None = None,
                             status: str | None = None) -> dict:
        """Atomically undo every edit of the open transaction and register it.

        status: 'rejected' (candidate turned down / explicit revert) or 'failed'
        (an exception interrupted the transaction). Defaults to 'rejected' when
        no error is given, 'failed' when one is.
        """
        if self.txn is None:
            return {"ok": True, "note": "no open transaction", "restored": [],
                    "undone": [], "remaining": []}
        try:
            rep = self.txn.restore_pristine(verify=True)
        except Exception as ex:
            # FAILED rollback: active tracking (self.applied, applied_here, ledger)
            # must remain UNCHANGED so no live edit ID is ever lost.
            self.txn.mark("failed", error=f"{error}; restore error: {ex}")
            self.ledger.register(self.txn, "failed", False, str(ex))
            self.txn = None
            raise
        # a clean restore still ends the transaction under its final status
        final_status = status or ("failed" if error else "rejected")
        undone = [e for e in self.txn.edit_ids() if e not in self._reverted_ids]
        self.txn.mark(final_status, error=error)
        self.ledger.register(self.txn, final_status, False, error)
        if rep["ok"]:
            # tracking is updated ONLY after the verified tensor restore
            self._reverted_ids.update(undone)
            self.applied = [a for a in self.applied
                            if a.get("edit_id") not in set(undone)]
            self._edit_order = [e for e in self._edit_order if e not in set(undone)]
            # NOTE: the transaction's durable status ('rejected'/'failed') already
            # covers its edits via txn.mark(); per-edit 'reverted' statuses are
            # only written by the explicit per-edit revert path (t_revert).
        out = {"ok": rep["ok"], "txn_id": rep["txn_id"], "status": final_status,
               "error": error, "undone": undone,
               "remaining": [a.get("edit_id") for a in self.applied],
               **rep}
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
        self.txn.record_edit(out["edit_id"], out["tensor"], {
            "edited_sha256": out["edited_sha256"], "base_sha256": out["base_sha256"],
            "shape": [int(x) for x in self.backend.shape_of(out["tensor"])],
            "dtype": self.backend.dtype_of(out["tensor"]),
            "changed_entries": out["changed_entries"],
            "relative_change": out["relative_change"],
        }, {"reason": reason, "hypothesis": hypothesis})
        self.applied.append(out)
        self._edit_order.append(out["edit_id"])
        return {"ok": True, **out, "txn_id": self.txn.txn_id,
                "n_edits_in_transaction": len(self.txn.edits)}

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

    # ------------------------------------------------------------- revert V3
    def _records(self) -> Dict[str, EditRecord]:
        """All edit records of this run, in execution order (open txn included).

        Records from the open transaction take precedence: they carry txn_id
        and the exact before/after hashes (editor.apply results do not).
        """
        recs: Dict[str, EditRecord] = {}
        if self.txn is not None and self.txn.edits:
            for e in self.txn.edits:
                recs[e["edit_id"]] = e
        for i, a in enumerate(self.applied):
            if a["edit_id"] in recs:
                continue
            r = EditRecord({
                "edit_id": a["edit_id"], "txn_id": a.get("txn_id"),
                "tensor": a["tensor"],
                "tensor_sha256_after": a["edited_sha256"],
                "tensor_sha256_before": a["base_sha256"],
                "txn_order": i,
            })
            recs[r["edit_id"]] = r
        return recs

    def _revert_one_edit(self, rec: EditRecord, verify: bool = True) -> str:
        """Restore ONE tensor to the pristine pre-image that existed before this
        edit was applied. This is the only code allowed to say 'undone'.

        The pre-image is the pristine bytes saved for that tensor before its
        FIRST edit in the run (per-transaction journal, promoted across
        transactions) - NOT the delta delta_store, which is forward-only.

        Returns the edit_id on success; raises with the original error details
        on any failure (no tracking is touched).
        """
        name = rec["tensor"]
        pre = self._pristine_preimages.get(name, {}).get(rec["tensor_sha256_before"])
        if pre is None and (self.txn is not None
                            and rec.get("txn_id") == self.txn.txn_id
                            and name in self.txn._saved
                            and self.txn._saved[name] == rec["tensor_sha256_before"]):
            # the edit lives in the OPEN transaction: its journal holds the
            # exact pre-transaction bits, which for a whole-txn revert are
            # precisely the pre-images needed
            pre = self.txn._preimage(name)
        if pre is None:
            raise RuntimeError(
                f"no pristine pre-image saved for tensor {name!r} "
                f"(base hash {rec['tensor_sha256_before'][:12]}); this edit cannot "
                "be individually reverted - use revert_transaction"
            )
        from ..sparse_deltas.cast import cast_to_dtype
        from ..sparse_deltas.store import sha256_arr
        if verify:
            h = sha256_arr(cast_to_dtype(pre, self.backend.dtype_of(name)))
            if h != rec["tensor_sha256_before"]:
                raise RuntimeError(
                    f"{name}: pristine pre-image hash mismatch ({h[:12]} != "
                    f"{rec['tensor_sha256_before'][:12]}); tensor NOT modified")
        self.backend.set_tensor(name, pre)
        if verify and self.backend.tensor_hash(name) != rec["tensor_sha256_before"]:
            raise RuntimeError(
                f"{name}: post-restore hash mismatch; rollback NOT verified")
        return rec["edit_id"]

    def _validate_partial_revert(self, target_recs: List[EditRecord]) -> None:
        """A partial revert must not break a SAME-TENSOR dependency.

        Per-edit restore works by pre-images, so an edit may only be reverted
        when every NEWER non-reverted edit on the SAME tensor has already been
        reverted (newest-first order guarantees this for n-based reverts). If a
        caller (e.g. a future out-of-order revert API) requests such an edit,
        reject it with a clear error and NO model modification.
        """
        recs = self._records()
        reverted = self._reverted_ids
        target_ids = {r["edit_id"] for r in target_recs}
        for rec in target_recs:
            for other in recs.values():
                if (other["tensor"] == rec["tensor"]
                        and other["edit_id"] != rec["edit_id"]
                        and other["edit_id"] not in target_ids
                        and other.get("txn_order", 0) > rec.get("txn_order", 0)
                        and other["edit_id"] not in reverted):
                    raise RevertConflict(
                        f"refusing to revert {rec['edit_id']}: newer edit "
                        f"{other['edit_id']} on the same tensor {rec['tensor']} is "
                        "still applied; revert it first (newest-first)"
                    )

    def t_revert(self, n: int = 1) -> dict:
        """Undo the last n APPLIED EDITS, newest first (n = number of edits).

        Each edit's tensor is restored to the exact pristine bits saved before
        that edit, and the SHA-256 is verified AFTER the write. Only then are
        the edit IDs reported as "undone" and removed from applied tracking.
        A failed restore raises - no ID is ever removed without a verified
        tensor restore, and a repeated revert is an idempotent no-op.

        Returns: {ok, undone: [edit_id], restored: [tensor names], remaining:
        [still-applied edit_ids], verified, txn_id, error}.

        To undo an ENTIRE transaction atomically, use revert_transaction().
        """
        n = max(1, int(n))
        recs = self._records()
        live = [rid for rid in self._edit_order if rid in recs and rid not in self._reverted_ids]
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
                # apply in reverse execution order so per-tensor pre-images compose
                undone.append(self._revert_one_edit(rec))
                restored.append(rec["tensor"])
        except RevertConflict as e:
            return {"ok": False, "error": str(e), "undone": [], "restored": [],
                    "remaining": live, "verified": False,
                    "txn_id": self.txn.txn_id if self.txn else None}
        except Exception as ex:
            # FAILED (possibly mid-way): report honestly, keep ALL tracking.
            # Already-restored tensors stay restored (their IDs removed), the
            # failing edit and everything else stay active. No live model state
            # is claimed clean unless every restore verified.
            done = set(undone)
            if done:
                self._reverted_ids.update(done)
                self.applied = [a for a in self.applied
                                if a.get("edit_id") not in done]
                self._edit_order = [e for e in self._edit_order if e not in done]
            remaining = [a.get("edit_id") for a in self.applied]
            return {"ok": False, "error": f"{type(ex).__name__}: {ex}",
                    "undone": undone, "restored": restored,
                    "remaining": remaining, "verified": False,
                    "txn_id": self.txn.txn_id if self.txn else self.last_txn_id}
        # verified restores only: now update every tracking surface at once
        self._reverted_ids.update(undone)
        undone_set = set(undone)
        self.applied = [a for a in self.applied
                        if a.get("edit_id") not in undone_set]
        self._edit_order = [e for e in self._edit_order if e not in undone_set]
        if self.txn is not None:
            self.txn.mark_edits_reverted(undone, None)
        if self.last_txn_id:
            self.ledger.mark_edits_reverted(self.last_txn_id, undone, None)
        return {"ok": True, "undone": undone, "restored": restored,
                "remaining": [a.get("edit_id") for a in self.applied],
                "verified": True,
                "txn_id": self.txn.txn_id if self.txn else self.last_txn_id,
                "error": None}

    def revert_transaction(self, txn_id: str | None = None) -> dict:
        """Atomically undo an ENTIRE transaction (all of its edits, verified).

        txn_id=None reverts the currently open transaction. For a committed
        transaction the pre-transaction pristine bits are restored from the
        journal promoted at commit time; the ledger and applied tracking are
        updated only after hash-verified restoration.
        """
        if txn_id is None or (self.txn is not None and self.txn.txn_id == txn_id
                              and self.txn.status == "open"):
            if self.txn is None:
                return {"ok": True, "undone": [], "restored": [], "remaining":
                        [a.get("edit_id") for a in self.applied], "verified": True,
                        "txn_id": None, "error": None,
                        "note": "no open transaction; idempotent no-op"}
            rep = self.rollback_transaction(status="rejected",
                                            error="explicit revert_transaction")
            return {"ok": rep["ok"], "undone": rep["undone"],
                    "restored": rep["restored"], "remaining": rep["remaining"],
                    "verified": rep["ok"], "txn_id": rep["txn_id"],
                    "error": rep.get("error"), **{k: v for k, v in rep.items()
                                                  if k not in ("ok", "undone", "restored",
                                                               "remaining", "verified", "txn_id",
                                                               "error")}}
        return {"ok": False, "error":
                f"transaction {txn_id!r} is closed and its per-transaction journal "
                "is no longer mounted; rebuild from the checkpoint chain instead "
                "with CheckpointManager.rebuild_chain_after_reset(backend, kept_ids)",
                "undone": [], "restored": [], "remaining":
                [a.get("edit_id") for a in self.applied], "verified": False,
                "txn_id": txn_id}

    def revert_last_n(self, n: int = 1) -> dict:
        """Loop-facing revert that always restores tensors (no open-txn required)."""
        return self.t_revert(n=n)

    def t_revert_transaction(self, txn_id: str | None = None) -> dict:
        """Tool-surface entry point for revert_transaction (see below)."""
        return self.revert_transaction(txn_id)

    def revert_whole_transaction(self, txn_id: str | None = None) -> dict:
        """Loop/agent-facing whole-transaction revert (alias of revert_transaction)."""
        return self.revert_transaction(txn_id)

    def register_loop_edit(self, applied_now: dict, hypothesis: str = "",
                           agent: str = "unknown") -> EditRecord:
        """Record a loop-applied edit (experiment/loop.py path) in the open
        transaction AND in applied tracking, so per-edit revert sees it."""
        rec = self.txn.record_edit(applied_now["edit_id"], applied_now["tensor"], {
            "edited_sha256": applied_now["edited_sha256"],
            "base_sha256": applied_now["base_sha256"],
            "shape": [int(x) for x in self.backend.shape_of(applied_now["tensor"])],
            "dtype": self.backend.dtype_of(applied_now["tensor"]),
            "changed_entries": applied_now["changed_entries"],
            "relative_change": applied_now["relative_change"],
        }, {"hypothesis": hypothesis, "agent": agent})
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
                    "exact verified tensor restore). Use revert_transaction to "
                    "undo a whole transaction atomically."},
            {"name": "revert_transaction", "args": {"txn_id": "str (optional)"},
             "doc": "Atomically undo an ENTIRE transaction: every edit of the "
                    "transaction, verified, in one operation."},
            {"name": "finish", "args": {"summary": "str"},
             "doc": "End the experiment with a summary."},
        ]

    def call(self, name: str, args: Dict) -> dict:
        fn: Callable | None = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name!r}", "available": [s["name"] for s in self.schema()]}
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
