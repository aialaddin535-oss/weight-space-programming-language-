"""Aladdin WSP command line interface.

    python -m aladdin.cli doctor
    python -m aladdin.cli model-info [--model-path DIR]
    python -m aladdin.cli verify-model
    python -m aladdin.cli inspect
    python -m aladdin.cli baseline --limit-per-category 2
    python -m aladdin.cli run --agent scripted --steps 3 --limit-per-category 2
    python -m aladdin.cli resume --experiment exp_xxx
    python -m aladdin.cli evaluate --split heldout --experiment exp_xxx
    python -m aladdin.cli edits list
    python -m aladdin.cli edits show --id e_xxx
    python -m aladdin.cli rollback --keep e_xxx
    python -m aladdin.cli report --experiment exp_xxx
    python -m aladdin.cli verify
    python -m aladdin.cli download            # the required WSP checkpoint
    python -m aladdin.cli download --repo-id RikiyaT/Qwen3.5-0.8B-weight-space-programming
    python -m aladdin.cli nvidia-models
    python -m aladdin.cli test
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .checkpoints.manager import CheckpointManager
from .config import PROJECT_ROOT, ensure_dirs, load_config, path_of, redacted
from .evaluation.datasets import load_splits
from .evaluation.runner import compare, run_eval
from .experiment.loop import run_experiment
from .experiment.state import ExperimentState
from .inspection.tensors import suggest_editable, tensor_stats
from .model.loader import (describe_model, download_model, load_model,
                           model_identity_report, verify_against_baseline,
                           write_or_check_baseline)
from .model_identity import REQUIRED_MODEL_ID
from .reporting import experiment_report, load_json, write_markdown
from .sparse_deltas.store import DeltaStore


def _cfg(args):
    cfg = load_config(args.config)
    if getattr(args, "backend", None):
        cfg["backend"]["kind"] = args.backend
    if getattr(args, "model_id", None):
        cfg["backend"]["model_id"] = args.model_id
    if getattr(args, "model_path", None):
        cfg["backend"]["model_path"] = args.model_path
    if getattr(args, "device", None):
        cfg["backend"]["device"] = args.device
    if getattr(args, "load_mode", None):
        cfg["backend"]["load_mode"] = args.load_mode
    if getattr(args, "agent", None):
        cfg["agent"]["provider"] = args.agent
    return cfg


def _factory(cfg):
    return lambda: load_model(cfg)[0]


# --------------------------------------------------------------------- commands
def cmd_doctor(args) -> int:
    cfg = _cfg(args)
    from .agent.nvidia_client import NvidiaClient

    cl = NvidiaClient(cfg)
    rep = {
        "project_root": str(PROJECT_ROOT),
        "config": cfg.get("_config_path") or "(defaults)",
        "python": sys.version.split()[0],
        "backend": cfg["backend"]["kind"],
        "required_model_id": cfg["backend"]["model_id"],
        "model_path": cfg["backend"]["model_path"] or "<unset>",
        "load_mode": cfg["backend"].get("load_mode"),
        "nvidia": cl.status(),
        "packages": {},
    }
    for mod in ("numpy", "requests", "torch", "transformers", "huggingface_hub",
                "safetensors", "pytest"):
        try:
            m = __import__(mod)
            rep["packages"][mod] = getattr(m, "__version__", "installed")
        except Exception:
            rep["packages"][mod] = None
    try:
        import torch
        rep["cuda"] = bool(torch.cuda.is_available())
        rep["cuda_devices"] = ([torch.cuda.get_device_name(i)
                                for i in range(torch.cuda.device_count())]
                               if rep["cuda"] else [])
    except Exception:
        rep["cuda"] = None
    if cfg["backend"]["model_path"]:
        rep["checkpoint"] = model_identity_report(cfg["backend"]["model_path"],
                                                  cfg["backend"]["model_id"])
    ensure_dirs(cfg)
    rep["paths"] = {k: str(path_of(cfg, k)) for k in cfg["paths"]}
    print(json.dumps(rep, indent=1))
    return 0


def cmd_model_info(args) -> int:
    cfg = _cfg(args)
    path = args.model_path or cfg["backend"]["model_path"]
    if not path:
        print(json.dumps({
            "required_model_id": cfg["backend"]["model_id"],
            "model_path": None,
            "hint": "run `aladdin download` and set ALADDIN_MODEL_PATH",
        }, indent=1))
        return 2
    rep = model_identity_report(path, cfg["backend"]["model_id"])
    print(json.dumps(rep, indent=1))
    return 0 if rep["identity_verified"] else 1


def cmd_verify_model(args) -> int:
    cfg = _cfg(args)
    backend, _ = load_model(cfg)
    verify = getattr(backend, "verify", None)
    if not callable(verify):
        print(json.dumps({"backend": backend.kind,
                          "verification": "not applicable (fixture backend)"},
                         indent=1))
        return 0
    rep = verify()
    print(json.dumps(rep, indent=1)[:8000])
    return 0 if rep.get("ok") else 1


def cmd_nvidia_models(args) -> int:
    cfg = _cfg(args)
    from .agent.nvidia_client import NvidiaClient

    cl = NvidiaClient(cfg)
    st = cl.status()
    print(json.dumps({"status": st}, indent=1))
    if not st["api_key_present"]:
        print("\nNVIDIA_API_KEY is not set - cannot list models. "
              "Verified public catalogue: see docs/NVIDIA_MODEL.md")
        return 2
    ids = cl.list_models()
    print(f"\n{len(ids)} models available to this key:")
    for i in ids:
        print(" ", i)
    if cfg["agent"]["model"]:
        print(f"\nConfigured NVIDIA_MODEL={cfg['agent']['model']} "
              f"present={cfg['agent']['model'] in ids}")
    return 0


def cmd_inspect(args) -> int:
    cfg = _cfg(args)
    backend, _ = load_model(cfg)
    desc = describe_model(backend)
    print(json.dumps(desc, indent=1))
    base = write_or_check_baseline(backend, cfg)
    print(f"\nbaseline manifest: {base['path']} "
          f"(created={base['created']}, pristine={base['pristine']})")
    rows = suggest_editable(backend, cfg, max_layers=args.layers)
    print("\neditable candidates:")
    for r in rows:
        print(f"  {r['name']:<55} {str(r['shape']):<18} entries={r['n_entries']:,}")
    if args.stats:
        print("\ntensor stats:")
        print(json.dumps(tensor_stats(backend, args.stats), indent=1)[:4000])
    return 0


def cmd_baseline(args) -> int:
    cfg = _cfg(args)
    backend, _ = load_model(cfg)
    dev, held = load_splits(cfg)
    ds = held if args.split == "heldout" else dev
    ensure_dirs(cfg)
    out = path_of(cfg, "logs") / f"baseline_{args.split}"
    res = run_eval(backend, ds, cfg, out_dir=out, label=f"baseline_{args.split}",
                   limit_per_category=args.limit_per_category)
    a = res["aggregate"]
    print(json.dumps({"split": args.split, "n": a["n"], "composite": a["composite"],
                      "categories": {k: v["mean"] for k, v in a["categories"].items()},
                      "out": str(out)}, indent=1))
    return 0


def cmd_run(args) -> int:
    cfg = _cfg(args)
    final = run_experiment(cfg, experiment=args.experiment, max_steps=args.steps,
                           agent_kind=args.agent, limit_per_category=args.limit_per_category,
                           resume=args.resume)
    print(json.dumps({k: v for k, v in final.items() if k != "state"}, indent=1)[:3000])
    return 0 if final.get("ok") else 1


def cmd_resume(args) -> int:
    cfg = _cfg(args)
    d = path_of(cfg, "experiments") / args.experiment
    st = ExperimentState.load(d / "state.json")
    print(json.dumps(st.summary(), indent=1))
    if not st.is_resumable():
        print("\nstate is not resumable (already finished or errored)")
        return 2
    final = run_experiment(cfg, experiment=args.experiment, max_steps=st.data["max_steps"],
                           agent_kind=args.agent, resume=True)
    print(json.dumps(final.get("state", final), indent=1))
    return 0 if final.get("ok") else 1


def cmd_evaluate(args) -> int:
    """Compare the required checkpoint BEFORE and AFTER kept WSP edits.

    The baseline is the freshly loaded, unedited snapshot; the edited state is
    base + kept sparse deltas, rebuilt and hash-verified. Use `--split heldout`
    only for the final report (the search must never see it).
    """
    cfg = _cfg(args)
    backend, _ = load_model(cfg)
    dev, held = load_splits(cfg)
    ds = held if args.split == "heldout" else dev
    exp_dir = path_of(cfg, "experiments") / args.experiment if args.experiment else None
    keep = []
    if exp_dir and (exp_dir / "state.json").exists():
        st = ExperimentState.load(exp_dir / "state.json")
        keep = st.data.get("kept_edits", [])
    ckpt = CheckpointManager(backend, path_of(cfg, "checkpoints"),
                             deltas_root=path_of(cfg, "deltas"), backend_factory=_factory(cfg))
    base_res = run_eval(backend, ds, cfg, out_dir=None, label="baseline_required_checkpoint")
    if keep:
        rep = ckpt.rebuild_chain_after_reset(backend, keep)
        if not rep["ok"]:
            print(json.dumps(rep, indent=1))
            return 1
    edit_res = run_eval(backend, ds, cfg, out_dir=None, label="edited_checkpoint")
    cmp = compare(base_res, edit_res, cfg)
    out = {"split": args.split, "n": base_res["aggregate"]["n"],
           "model_id": cfg["backend"]["model_id"],
           "baseline": base_res["aggregate"]["composite"],
           "edited": edit_res["aggregate"]["composite"],
           "delta": cmp["delta"], "per_category": cmp["per_category"],
           "kept_edits": keep}
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"baseline": base_res, "edited": edit_res, "compare": cmp}, f,
                      indent=1, ensure_ascii=False)
        print(f"written: {args.out}")
    return 0


def cmd_compare(args) -> int:
    cfg = _cfg(args)
    a, b = load_json(args.a), load_json(args.b)
    print(json.dumps(compare(a, b, cfg), indent=1))
    return 0


def cmd_edits(args) -> int:
    cfg = _cfg(args)
    store = DeltaStore(path_of(cfg, "deltas"))
    if args.action == "list":
        rows = store.list_edits()
        print(f"{len(rows)} edit(s) in {store.root}\n")
        for r in rows:
            print(f"{r['edit_id']:<28} {r['status']:<10} tensors={r['n_tensors']} "
                  f"entries={r['changed_entries']} rel={r['relative_change']:.4f}")
            print(f"    {r['tensors']}")
            print(f"    {(r['reason'] or '')[:150]}")
        return 0
    man, arrays = store.load(args.id)
    print(json.dumps(man, indent=1, ensure_ascii=False))
    print(f"\narrays: { {k: list(v.shape) for k, v in arrays.items()} }")
    return 0


def cmd_rollback(args) -> int:
    cfg = _cfg(args)
    backend, _ = load_model(cfg)
    ckpt = CheckpointManager(backend, path_of(cfg, "checkpoints"),
                             deltas_root=path_of(cfg, "deltas"), backend_factory=_factory(cfg))
    keep = args.keep or []
    if args.experiment:
        st = ExperimentState.load(path_of(cfg, "experiments") / args.experiment / "state.json")
        keep = st.data.get("kept_edits", [])
    rep = ckpt.rebuild_chain_after_reset(backend, keep)
    rep["kept"] = keep
    print(json.dumps(rep, indent=1)[:3000])
    return 0 if rep.get("ok") else 1


def cmd_verify(args) -> int:
    cfg = _cfg(args)
    backend, _ = load_model(cfg)
    print(json.dumps(verify_against_baseline(backend, cfg), indent=1))
    return 0


def cmd_report(args) -> int:
    cfg = _cfg(args)
    d = path_of(cfg, "experiments") / args.experiment
    text = experiment_report(d)
    out = d / "report.md"
    write_markdown(text, out)
    print(text)
    print(f"\nwritten: {out}")
    return 0


def cmd_download(args) -> int:
    model_id = args.repo_id or REQUIRED_MODEL_ID
    dest = args.dest or str(path_of(_cfg(args), "models") / model_id.split("/")[-1])
    print(f"downloading {model_id} -> {dest}")
    if "Qwen/Qwen3.5-0.8B" == model_id:
        print("WARNING: that is the BASE model. Aladdin WSP requires the edited "
              f"checkpoint {REQUIRED_MODEL_ID}.")
    rep = download_model(model_id, dest, revision=args.revision or "")
    print(json.dumps(rep, indent=1))
    print(f"\npoint the project at it:\n  export ALADDIN_MODEL_PATH={rep['path']}\n"
          "  export ALADDIN_BACKEND=hf_torch")
    return 0


def cmd_test(args) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", str(PROJECT_ROOT / "tests")]
    if args.verbose:
        cmd.insert(3, "-vv")
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aladdin",
                                description="Weight-space programming for Hindi/Hinglish")
    p.add_argument("--config", default=os.environ.get("ALADDIN_CONFIG") or
                   str(PROJECT_ROOT / "config" / "default.json"))
    p.add_argument("--backend", choices=["tiny_numpy", "hf_torch"])
    p.add_argument("--model-id")
    p.add_argument("--model-path")
    p.add_argument("--device")
    p.add_argument("--load-mode", choices=["text", "multimodal"])
    p.add_argument("--agent", choices=["nvidia", "scripted"])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sub.add_parser("verify-model").set_defaults(fn=cmd_verify_model)
    s = sub.add_parser("model-info"); s.add_argument("--model-path")
    s.set_defaults(fn=cmd_model_info)
    sub.add_parser("nvidia-models").set_defaults(fn=cmd_nvidia_models)

    s = sub.add_parser("inspect"); s.add_argument("--stats"); s.add_argument("--layers", type=int, default=6)
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("baseline"); s.add_argument("--split", default="dev", choices=["dev", "heldout"])
    s.add_argument("--limit-per-category", type=int, default=0); s.set_defaults(fn=cmd_baseline)

    s = sub.add_parser("run"); s.add_argument("--experiment"); s.add_argument("--steps", type=int)
    s.add_argument("--limit-per-category", type=int, default=0)
    s.add_argument("--agent", choices=["nvidia", "scripted"])
    s.add_argument("--resume", action="store_true"); s.set_defaults(fn=cmd_run)

    s = sub.add_parser("resume"); s.add_argument("--experiment", required=True)
    s.add_argument("--agent", choices=["nvidia", "scripted"]); s.set_defaults(fn=cmd_resume)

    s = sub.add_parser("evaluate"); s.add_argument("--split", default="heldout", choices=["dev", "heldout"])
    s.add_argument("--experiment"); s.add_argument("--out"); s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("compare"); s.add_argument("--a", required=True); s.add_argument("--b", required=True)
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("edits"); s.add_argument("action", choices=["list", "show"])
    s.add_argument("--id"); s.set_defaults(fn=cmd_edits)

    s = sub.add_parser("rollback"); s.add_argument("--experiment"); s.add_argument("--keep", nargs="*")
    s.set_defaults(fn=cmd_rollback)

    s = sub.add_parser("report"); s.add_argument("--experiment", required=True); s.set_defaults(fn=cmd_report)

    s = sub.add_parser("download")
    s.add_argument("--repo-id", default=REQUIRED_MODEL_ID,
                   help="HF repository id (default: the required WSP checkpoint)")
    s.add_argument("--dest"); s.add_argument("--revision"); s.set_defaults(fn=cmd_download)

    s = sub.add_parser("test"); s.add_argument("--verbose", action="store_true"); s.set_defaults(fn=cmd_test)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
