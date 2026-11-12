"""The experiment loop: baseline -> hypothesis -> edit -> evaluate -> accept/reject.

One candidate per step, always measured on the DEV split against the running
baseline, always rolled back on rejection by rebuilding the model from the base
weights plus the kept deltas. The held-out split is never touched here.

Every step writes: experiments/<id>/step_NNN.json, an append-only
experiments/<id>/history.jsonl, the state file, and the delta manifest.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List

from ..checkpoints.manager import CheckpointManager
from ..config import ensure_dirs, path_of, redacted
from ..evaluation.datasets import assert_no_heldout_leak, load_splits
from ..evaluation.runner import compare, run_eval
from ..forward_pass.engine import ForwardPassEngine
from ..inspection.tensors import suggest_editable
from ..model.loader import (describe_model, load_model, verify_against_baseline,
                            write_or_check_baseline)
from ..sparse_deltas.store import DeltaStore
from ..tensor_editor.engine import EditRefused, TensorEditor
from ..agent.tools import ToolRuntime
from .state import ExperimentState


def _failures_from(results: dict, limit: int = 8) -> List[dict]:
    items = sorted(results["items"], key=lambda s: s["composite"])[:limit]
    out = []
    for s in items:
        out.append({"id": s["id"], "category": s["category"], "score": s["composite"],
                    "notes": s["notes"][:4], "output_head": s["output"][:160]})
    return out


def _factory(cfg: dict) -> Callable[[], object]:
    """Pristine-weight loader used by the checkpoint manager for large models."""
    return lambda: load_model(cfg)[0]


def run_experiment(cfg: dict, experiment: str | None = None, max_steps: int | None = None,
                   agent_kind: str | None = None, limit_per_category: int = 0,
                   resume: bool = False, judge=None, verbose: bool = True) -> dict:
    ensure_dirs(cfg)
    experiment = experiment or time.strftime("exp_%Y%m%d-%H%M%S")
    exp_dir = path_of(cfg, "experiments") / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    state_path = exp_dir / "state.json"

    backend, model_desc = load_model(cfg)
    model_desc = describe_model(backend)
    base_check = write_or_check_baseline(backend, cfg)
    if base_check.get("drift"):
        raise RuntimeError(
            "the model's tensors differ from the recorded baseline: "
            f"{base_check['drift'][:5]} - refusing to run on modified weights")

    ckpt = CheckpointManager(backend, path_of(cfg, "checkpoints"),
                             deltas_root=path_of(cfg, "deltas"),
                             backend_factory=_factory(cfg))
    store = DeltaStore(path_of(cfg, "deltas"))
    editor = TensorEditor(backend, cfg, store)
    fwd = ForwardPassEngine(backend, budget=int(cfg["forward_pass"]["budget"]), run_dir=exp_dir)

    dev, held = load_splits(cfg)

    if resume and state_path.exists():
        state = ExperimentState.load(state_path)
        if not state.is_resumable():
            return {"ok": False, "reason": "state is not resumable",
                    "summary": state.summary()}
        # replay kept deltas onto the freshly loaded base model
        rep = ckpt.rebuild_chain(state.data["kept_edits"])
        if not rep["ok"]:
            return {"ok": False, "reason": f"could not replay kept deltas: {rep['failed']}"}
        if verbose:
            print(f"[resume] step {state.data['step']}, replayed {len(state.data['kept_edits'])} deltas")
    else:
        state = ExperimentState.create(
            state_path, experiment=experiment, backend=backend.kind,
            model_desc=model_desc, config_path=cfg.get("_config_path", ""),
            max_steps=int(max_steps or cfg["agent"].get("max_steps", 14)))
        state.data["config_snapshot"] = redacted(cfg)
        state.save()

    max_steps = int(max_steps or state.data["max_steps"])

    # ------------------------------------------------------------- baseline
    baseline_path = exp_dir / "baseline_dev.json"
    if baseline_path.exists():
        baseline = json.load(open(baseline_path, "r", encoding="utf-8"))
        if verbose:
            print(f"[baseline] reusing {baseline['aggregate']['composite']}")
    else:
        baseline = run_eval(backend, dev, cfg, out_dir=exp_dir / "eval_baseline",
                            label="baseline", limit_per_category=limit_per_category,
                            judge=judge)
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=1, ensure_ascii=False)
        if verbose:
            print(f"[baseline] dev composite = {baseline['aggregate']['composite']}")
    assert_no_heldout_leak({s["id"] for s in baseline["items"]}, held)
    state.record_baseline(baseline["aggregate"]["composite"])
    ckpt.save_base()

    # ---------------------------------------------------------------- agent
    runtime = ToolRuntime(backend, cfg, editor, fwd, dev, exp_dir,
                          baseline_result=baseline, task=experiment)
    if agent_kind:
        cfg["agent"]["provider"] = agent_kind
    from ..agent import build_agent

    agent = build_agent(cfg, runtime, exp_dir)

    # ----------------------------------------------------------------- loop
    steps_done = 0
    start_step = int(state.data["step"])
    for step in range(start_step, max_steps):
        if verbose:
            print(f"\n=== step {step + 1}/{max_steps} ===")
        failures = _failures_from(baseline)
        try:
            out = agent.run_step(step=step, failures=failures,
                                 baseline_composite=baseline["aggregate"]["composite"],
                                 model_desc=model_desc, budget_left=fwd.remaining())
        except Exception as e:
            # an agent crash mid-interaction must not leave partially edited weights
            rb = runtime.rollback_transaction(error=repr(e))
            state.record_step({"step": step, "error": repr(e), "accepted": False,
                               "transaction_rollback": rb})
            state.finish("error")
            return {"ok": False, "error": repr(e), "transaction_rollback": rb,
                    "state": state.summary()}

        op = out.get("operation")
        applied_ids: List[str] = list(out.get("applied") or [])
        applied_now = None

        # edits the agent applied through its own tool calls live in an open
        # transaction; if the agent ALSO returned a direct operation, close the
        # tool transaction first so the two paths never mix in one transaction
        if runtime.txn is not None and runtime.txn.status == "open":
            if applied_ids:
                runtime.commit_transaction()
            else:
                applied_ids = list(runtime.txn.edit_ids())

        if op and not applied_ids:
            runtime.begin_transaction()
            try:
                runtime.txn.capture_pristine(op.get("tensor") or op.get("name"))
                applied_now = editor.apply(op, reason=out.get("hypothesis", "") or op.get("reason", ""),
                                           meta={"hypothesis": out.get("hypothesis", ""),
                                                 "experiment": experiment,
                                                 "txn_id": runtime.txn.txn_id,
                                                 "agent": getattr(agent, "name", "unknown")})
                # one registration path for BOTH the transaction journal and the
                # runtime's applied tracking (kept in sync for per-edit revert)
                runtime.register_loop_edit(applied_now,
                                           hypothesis=out.get("hypothesis", ""),
                                           agent=getattr(agent, "name", "unknown"))
                applied_ids = [applied_now["edit_id"]]
            except EditRefused as e:
                runtime.rollback_transaction(error=str(e))
                entry = {"step": step, "refused": str(e), "accepted": False,
                         "operation": op, "candidate_composite": None}
                state.record_step(entry)
                if verbose:
                    print(f"[refused] {e}")
                continue

        if not applied_ids:
            entry = {"step": step, "no_edit": True, "accepted": False,
                     "agent_summary": out.get("summary", "")}
            state.record_step(entry)
            if verbose:
                print(f"[agent] no edit proposed: {out.get('summary', '')[:100]}")
            if out.get("summary") and "exhausted" in str(out.get("summary")):
                break
            continue

        # ------------------------------------------------------- evaluation
        cand = run_eval(backend, dev, cfg, out_dir=exp_dir / f"eval_step_{step:03d}",
                        label=f"candidate_step{step:03d}",
                        limit_per_category=limit_per_category, judge=judge)
        assert_no_heldout_leak({s["id"] for s in cand["items"]}, held)
        decision = compare(baseline, cand, cfg)
        accepted = bool(decision["accept"])

        txn_id = runtime.txn.txn_id if runtime.txn is not None else runtime.last_txn_id
        entry = {
            "step": step, "edit_id": applied_ids[-1], "edit_ids": list(applied_ids),
            "txn_id": txn_id, "n_edits": len(applied_ids), "operation": op,
            "hypothesis": out.get("hypothesis", ""),
            "agent_summary": out.get("summary", ""),
            "baseline_composite": baseline["aggregate"]["composite"],
            "candidate_composite": cand["aggregate"]["composite"],
            "delta": decision["delta"], "accepted": accepted,
            "reason": decision["reason"],
            "regressions": decision["regressions"],
            "per_category": decision["per_category"],
            "evidence": out.get("evidence", {}),
        }
        with open(exp_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with open(exp_dir / f"step_{step:03d}.json", "w", encoding="utf-8") as f:
            json.dump({"entry": entry, "decision": decision,
                       "candidate_aggregate": cand["aggregate"]}, f, indent=1,
                      ensure_ascii=False)

        if accepted:
            for eid in applied_ids:            # EVERY edit of the group is accepted
                store.update_status(eid, "accepted",
                                    {"evaluation": {"composite": cand["aggregate"]["composite"],
                                                    "delta": decision["delta"]},
                                     "txn_id": txn_id})
            if runtime.txn is not None:
                runtime.commit_transaction()
            baseline = cand                    # running best becomes the new baseline
            ckpt.save(f"best_step{step:03d}", note=entry["reason"], txn=txn_id,
                      parent=state.data["kept_edits"][-1] if state.data["kept_edits"] else "base",
                      metrics={"dev_composite": cand["aggregate"]["composite"]})
            if verbose:
                print(f"[accept +{decision['delta']:.4f}] {len(applied_ids)} edit(s) "
                      f"{applied_ids[0]}..{applied_ids[-1]}")
        else:
            for eid in applied_ids:            # EVERY edit of the group is rejected
                store.update_status(eid, "rejected",
                                    {"evaluation": {"composite": cand["aggregate"]["composite"],
                                                    "delta": decision["delta"]},
                                     "rejection_reason": decision["reason"],
                                     "txn_id": txn_id})
            # roll back the WHOLE transaction: exact pristine bits of every touched
            # tensor, hash-verified; then rebuild base + kept deltas as the
            # independent checkpoint-chain path and cross-check consistency
            rb = runtime.rollback_transaction(error=decision["reason"], status="rejected")
            rep = ckpt.rebuild_chain_after_reset(backend, state.data["kept_edits"])
            rep["transaction_rollback"] = rb
            rep["ok"] = bool(rep.get("ok")) and bool(rb.get("ok"))
            if not rep["ok"]:
                state.record_step(entry)
                state.finish("rollback_failed")
                return {"ok": False, "rollback": rep, "state": state.summary()}
            if verbose:
                print(f"[reject {decision['delta']:+.4f}] rolled back {len(applied_ids)} "
                      f"edit(s) - {decision['reason']}")

        state.record_step(entry)
        steps_done += 1

    state.finish("finished")
    final = {
        "ok": True,
        "experiment": experiment,
        "dir": str(exp_dir),
        "state": state.summary(),
        "forward_passes": fwd.summary(),
        "best_dev_composite": state.data["best_composite"],
        "baseline_dev_composite": state.data["baseline_composite"],
        "kept_edits": state.data["kept_edits"],
        "verify": verify_against_baseline(backend, cfg),
    }
    with open(exp_dir / "final.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=1, ensure_ascii=False)
    if verbose:
        print("\n" + json.dumps(final["state"], indent=1))
    return final
