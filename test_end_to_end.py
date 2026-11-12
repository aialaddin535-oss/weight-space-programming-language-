"""End-to-end: baseline -> scripted candidate -> accept/reject -> rollback -> resume."""
import json
from pathlib import Path

import pytest

from aladdin.checkpoints.manager import CheckpointManager
from aladdin.evaluation.datasets import Dataset
from aladdin.experiment.loop import run_experiment


def _write_datasets(root: Path):
    d = root / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (cat, prompt) in enumerate([
        ("roman_hinglish", "mujhe samaj nahi aya simple me batao"),
        ("devanagari_hindi", "मुझे समझ नहीं आया"),
        ("mixed_code_switch", "mera laptop slow chal raha hai kya karu"),
        ("instruction_following", "2 bullet me batao python kyu"),
    ]):
        rows.append({"id": f"d{i}", "category": cat, "prompt": prompt,
                     "expect_script": "either", "must_include": [["a", "e", "i", "o", "u"]],
                     "answer_tokens": [], "meaning": "test"})
    (d / "dev.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    held = [{"id": f"h{i}", "category": c, "prompt": p + " heldout",
             "expect_script": "either", "must_include": [], "answer_tokens": []}
            for i, (c, p) in enumerate([("factuality", "bharat ki rajdhani"),
                                        ("question_answering", "2+2 kitna")])]
    (d / "heldout.jsonl").write_text("\n".join(json.dumps(r) for r in held), encoding="utf-8")


@pytest.fixture()
def project(tmp_path, cfg):
    c = dict(cfg)
    c["paths"] = {k: str(tmp_path / v) for k, v in cfg["paths"].items()}
    c["_project_root"] = str(tmp_path)
    for p in c["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)
    c["forward_pass"] = dict(c["forward_pass"], budget=6)
    c["agent"] = dict(c["agent"], provider="scripted", max_steps=2)
    c["evaluation"] = dict(c["evaluation"], max_new_tokens=4)
    c["acceptance"] = dict(c["acceptance"], min_composite_gain=-1.0)   # accept everything
    _write_datasets(tmp_path)
    return c


def test_full_experiment_runs_and_is_auditable(project):
    final = run_experiment(project, experiment="e2e", max_steps=2, agent_kind="scripted",
                           verbose=False)
    assert final["ok"], final
    exp = Path(final["dir"])
    for f in ("state.json", "baseline_dev.json", "history.jsonl", "final.json"):
        assert (exp / f).exists(), f
    hist = [json.loads(l) for l in (exp / "history.jsonl").read_text().splitlines() if l.strip()]
    assert hist, "history must record each candidate"
    for h in hist:
        assert "delta" in h and "accepted" in h and "reason" in h
    st = json.load(open(exp / "state.json"))
    assert st["status"] == "finished"
    assert st["applied_total"] >= 1


def test_rejected_edit_is_rolled_back(project, cfg):
    c = dict(project)
    c["acceptance"] = dict(cfg["acceptance"], min_composite_gain=99.0)   # reject everything
    final = run_experiment(c, experiment="e2e_reject", max_steps=1, agent_kind="scripted",
                           verbose=False)
    assert final["ok"], final
    assert final["state"]["kept"] == 0 and final["state"]["rejected"] >= 1
    assert final["verify"]["ok"] is True, "weights must be back to the pristine baseline"


def test_resume_continues_from_state(project):
    """Simulate an interrupted run, then resume it and check it continues."""
    first = run_experiment(project, experiment="e2e_resume", max_steps=1, agent_kind="scripted",
                           verbose=False)
    assert first["ok"]
    exp = Path(first["dir"])
    # make the state look like a run that was killed after step 1 with budget left
    st = json.load(open(exp / "state.json"))
    st["status"] = "interrupted"
    st["max_steps"] = 2
    json.dump(st, open(exp / "state.json", "w"))

    second = run_experiment(project, experiment="e2e_resume", max_steps=2,
                            agent_kind="scripted", resume=True, verbose=False)
    assert second["ok"], second
    st2 = json.load(open(exp / "state.json"))
    assert st2["step"] == 2, st2
    assert st2["status"] == "finished"


def test_baseline_manifest_detects_drift(project, cfg):
    from aladdin.model.loader import load_model, verify_against_baseline, write_or_check_baseline

    backend, _ = load_model(project)
    write_or_check_baseline(backend, project)
    assert verify_against_baseline(backend, project)["ok"] is True
    n = "model.layers.0.mlp.down_proj.weight"
    w = backend.get_tensor(n)
    w[0, 0] += 1.0
    backend.set_tensor(n, w)
    r = verify_against_baseline(backend, project)
    assert r["ok"] is False and n in r["changed_tensors"]
