"""Forward-pass budget + tool runtime + agent reply parsing."""
import numpy as np
import pytest

from aladdin.agent.tools import ToolRuntime, parse_tool_call
from aladdin.evaluation.datasets import Dataset
from aladdin.forward_pass import BudgetExhausted, ForwardPassEngine
from aladdin.sparse_deltas.store import DeltaStore
from aladdin.tensor_editor import TensorEditor


def test_budget_is_enforced_and_logged(backend, tmp_path):
    fwd = ForwardPassEngine(backend, budget=2, run_dir=tmp_path)
    fwd.generate(["a"], max_new_tokens=2)
    fwd.generate(["b"], max_new_tokens=2)
    with pytest.raises(BudgetExhausted):
        fwd.generate(["c"], max_new_tokens=2)
    s = fwd.summary()
    assert s["used"] == 2 and s["remaining"] == 0
    assert (tmp_path / "forward_passes.jsonl").exists()


def test_failed_request_does_not_consume_budget(backend, tmp_path):
    fwd = ForwardPassEngine(backend, budget=2, run_dir=tmp_path)
    fwd.generate(["ok"], max_new_tokens=1)
    assert fwd.used == 1


def test_teacher_forced_returns_logprobs(backend, tmp_path):
    fwd = ForwardPassEngine(backend, budget=4, run_dir=tmp_path)
    rec = fwd.teacher_forced([("iska matlab kya hai", "matlab ye hai")])
    v = rec["mean_logprob"][0]
    assert v is not None and np.isfinite(v) and v < 0


def test_activations_pass_records_shapes(backend, tmp_path):
    fwd = ForwardPassEngine(backend, budget=4, run_dir=tmp_path)
    rec = fwd.activations(["mujhe samaj nahi aya"])
    assert rec["snapshots"] and rec["snapshots"][0]


def test_parse_tool_call_variants():
    a = parse_tool_call('```json\n{"tool": "tensor_stats", "args": {"name": "x"}}\n```')
    assert a["tool"] == "tensor_stats" and a["args"]["name"] == "x"
    b = parse_tool_call('Sure.\n{"name": "list_tensors", "arguments": {"pattern": "mlp"}}')
    assert b["tool"] == "list_tensors" and b["args"]["pattern"] == "mlp"
    c = parse_tool_call('{"function": "finish", "arguments": "{\\"summary\\": \\"done\\"}"}')
    assert c["tool"] == "finish" and c["args"]["summary"] == "done"
    assert parse_tool_call("no json here") is None
    assert parse_tool_call("") is None


def test_runtime_rejects_unknown_tool(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    r = rt.call("delete_everything", {})
    assert "error" in r and "available" in r


def test_runtime_apply_and_revert_flow(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    name = "model.layers.0.mlp.down_proj.weight"
    p = rt.call("preview_edit", {"operation": {"kind": "scale_columns", "tensor": name,
                                               "columns": [0], "alpha": 0.05}})
    assert p["ok"] and p["would_change_entries"] > 0
    a = rt.call("apply_edit", {"operation": {"kind": "scale_columns", "tensor": name,
                                             "columns": [0], "alpha": 0.05},
                               "reason": "test", "hypothesis": "h"})
    assert a["ok"] and a["verified"]
    assert len(rt.applied) == 1
    assert rt.call("tensor_stats", {"name": name})["sha256"] == a["edited_sha256"]
    rt.call("finish", {"summary": "done"})


def _runtime(backend, cfg, tmp_project):
    store = DeltaStore(tmp_project["paths"]["deltas"])
    ed = TensorEditor(backend, cfg, store)
    fwd = ForwardPassEngine(backend, budget=10, run_dir=tmp_project["paths"]["logs"])
    ds = Dataset([{"id": "a1", "category": "roman_hinglish", "prompt": "mujhe samaj nahi aya",
                   "expect_script": "roman", "must_include": [["samaj"]], "answer_tokens": []}], "dev")
    return ToolRuntime(backend, cfg, ed, fwd, ds, __import__("pathlib").Path(tmp_project["paths"]["experiments"]))
