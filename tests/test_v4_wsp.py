"""V4 regression tests.

Groups:
  A. required checkpoint identity and the text-only Qwen3.5 loading plan
     (against the REAL config.json / weight-key index of the published model);
  B. robust model verification (fake objects must not crash it);
  C. cross-transaction ledger ownership (the reported bug);
  D. committed-transaction revert, dependency handling, atomic failure recovery,
     restart persistence;
  E. NVIDIA agent synchronization; F. Hindi/Hinglish evaluation coverage.
"""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aladdin.agent.tools import RevertConflict, ToolRuntime
from aladdin.backends import qwen_loader as ql
from aladdin.config import DEFAULTS
from aladdin.edit_ledger import EditLedger
from aladdin.forward_pass import ForwardPassEngine
from aladdin.model_identity import (EXPECTED_HIDDEN_SIZE, EXPECTED_NUM_HIDDEN_LAYERS,
                                    EXPECTED_VOCAB_SIZE, KNOWN_MODEL_REVISION,
                                    REQUIRED_MODEL_ID)
from aladdin.sparse_deltas.store import DeltaStore
from aladdin.tensor_editor import TensorEditor

FIX = Path(__file__).resolve().parent / "fixtures"
N1 = "model.layers.0.mlp.down_proj.weight"
N2 = "model.layers.1.mlp.down_proj.weight"


# --------------------------------------------------------------------- helpers
def _op(name, cols, alpha=0.1, kind="scale_columns"):
    return {"kind": kind, "tensor": name, "columns": cols, "alpha": alpha}


def _runtime(backend, cfg, tmp_project, dataset=None):
    store = DeltaStore(tmp_project["paths"]["deltas"])
    ed = TensorEditor(backend, cfg, store)
    fwd = ForwardPassEngine(backend, budget=20, run_dir=tmp_project["paths"]["logs"])
    return ToolRuntime(backend, cfg, ed, fwd, dataset,
                       Path(tmp_project["paths"]["experiments"]))


def _fixture_keys():
    return json.load(open(FIX / "rikiya_weight_keys.json"))["keys"]


def _fixture_config():
    return json.load(open(FIX / "rikiya_config.json"))


def _make_snapshot(tmp_path, repo_id=REQUIRED_MODEL_ID, write_index=True,
                   metadata=True):
    """A snapshot-shaped directory with the REAL config and key index."""
    d = tmp_path / "snapshot"
    d.mkdir(exist_ok=True)
    cfg = _fixture_config()
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if write_index:
        keys = _fixture_keys()
        wm = {k: "model.safetensors-00001-of-00001.safetensors" for k in keys}
        (d / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 1746882752}, "weight_map": wm}),
            encoding="utf-8")
    if metadata:
        (d / "aladdin_checkpoint.json").write_text(json.dumps({
            "repo_id": repo_id, "revision": "main", "sha": KNOWN_MODEL_REVISION,
            "downloaded_utc": "2026-11-12T00:00:00Z",
            "is_required_wsp_checkpoint": repo_id == REQUIRED_MODEL_ID}),
            encoding="utf-8")
    return d


# ============================================ A. required checkpoint identity
def test_required_model_is_the_only_default():
    import aladdin
    from aladdin.cli import build_parser
    assert aladdin.REQUIRED_MODEL_ID == REQUIRED_MODEL_ID
    assert REQUIRED_MODEL_ID == "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
    assert aladdin.DEFAULT_MODEL_ID == REQUIRED_MODEL_ID
    assert DEFAULTS["backend"]["model_id"] == REQUIRED_MODEL_ID
    ns = build_parser().parse_args(["download"])
    assert ns.repo_id == REQUIRED_MODEL_ID
    # every shipped config names the edited checkpoint
    for p in (Path(__file__).resolve().parent.parent / "config").glob("*.json"):
        cfg = json.load(open(p))
        mid = cfg.get("backend", {}).get("model_id")
        assert mid == REQUIRED_MODEL_ID, f"{p.name} defaults to {mid!r}"



def test_real_config_geometry_is_detected(tmp_path):
    snap = _make_snapshot(tmp_path)
    a = ql.detect_architecture(snap)
    assert a["is_qwen3_5"] is True
    assert a["is_multimodal"] is True               # the checkpoint has a vision tower
    assert a["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert a["text_model_type"] == "qwen3_5_text"
    assert a["text_hidden_size"] == EXPECTED_HIDDEN_SIZE == 1024
    assert a["text_num_hidden_layers"] == EXPECTED_NUM_HIDDEN_LAYERS == 24
    assert a["text_vocab_size"] == EXPECTED_VOCAB_SIZE == 248320
    assert a["text_intermediate_size"] == 3584
    assert a["tie_word_embeddings"] is True
    assert a["identity_verified"] is True
    assert a["revision"] == KNOWN_MODEL_REVISION


def test_identity_check_rejects_the_base_model(tmp_path):
    snap = _make_snapshot(tmp_path, repo_id="Qwen/Qwen3.5-0.8B")
    ident = ql.check_checkpoint_identity(snap)
    assert ident["ok"] is False
    assert ident["identity_verified"] is False
    assert any("requires" in p for p in ident["problems"])


def test_text_only_plan_uses_the_real_weight_keys():
    """The published checkpoint stores text weights under model.language_model.*."""
    a = ql.detect_architecture(_make_snapshot(Path(_tmpdir())))
    plan = ql.plan_text_only_load(_fixture_keys(), a)
    assert plan["n_mapped"] == 320
    assert plan["skipped"]["vision"] == 153
    assert plan["skipped"]["mtp"] == 15
    assert plan["skipped"]["other"] == 0
    mapping = plan["mapping"]
    assert "model.embed_tokens.weight" in mapping
    assert "model.norm.weight" in mapping
    assert "model.layers.0.mlp.down_proj.weight" in mapping
    assert "model.layers.23.self_attn.o_proj.weight" in mapping
    assert mapping["model.embed_tokens.weight"] == (
        "model.language_model.embed_tokens.weight")
    assert not any(k.startswith(("model.visual.", "mtp.")) for k in mapping)
    assert not any(v.startswith(("model.visual.", "mtp.")) for v in mapping.values())
    # tied embeddings: the checkpoint has no lm_head and that is the only allowed miss
    assert "lm_head.weight" not in mapping
    assert plan["allowed_missing"] == ["lm_head.weight"]
    layers = {k.split(".layers.")[1].split(".")[0] for k in mapping if ".layers." in k}
    assert layers == {str(i) for i in range(24)}


def _tmpdir():
    import tempfile
    d = tempfile.mkdtemp()
    return d


def test_loader_selects_official_text_class_not_multimodal(tmp_path):
    a = ql.detect_architecture(_make_snapshot(tmp_path))
    cls, reason = ql._auto_class_for(a)
    assert cls == ql.TEXT_ONLY_CLASS == "Qwen3_5ForCausalLM"
    assert "AutoModelForImageTextToText is NOT used" in reason
    mcls, _ = ql._auto_class_for(a, mode="multimodal")
    assert mcls == "AutoModelForImageTextToText"


def test_loader_version_and_class_guards(monkeypatch, tmp_path):
    monkeypatch.setattr(ql, "transformers_version", lambda: "4.46.0")
    r = ql.check_transformers_version()
    assert r["ok"] is False and "pip install" in r["error"]
    with pytest.raises(ql.LoaderCompatibilityError, match="too old"):
        ql.load_qwen35(_make_snapshot(tmp_path))

    assert ql.TEXT_ONLY_CLASS == "Qwen3_5ForCausalLM"
    assert ql.TEXT_CONFIG_CLASS == "Qwen3_5TextConfig"


def test_missing_model_path_error_names_the_required_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        ql.detect_architecture(tmp_path / "nope")


# ================================================= B. robust verification
def _fake_model(n_layers=24, hidden=1024, vocab=248320, cls_name="Qwen3_5ForCausalLM",
                with_modules=True):
    params = {}

    def add(name, shape):
        params[name] = SimpleNamespace(shape=shape, dtype="bf16",
                                       numel=lambda s=shape: int(np.prod(s)),
                                       device="cpu")

    add("model.embed_tokens.weight", (vocab, hidden))
    for i in range(n_layers):
        add(f"model.layers.{i}.mlp.down_proj.weight", (hidden, 3584))
    add("model.norm.weight", (hidden,))
    ns = {
        "config": SimpleNamespace(hidden_size=hidden, num_hidden_layers=n_layers,
                                  vocab_size=vocab, tie_word_embeddings=True),
        "named_parameters": staticmethod(lambda: iter(params.items())),
        "named_buffers": staticmethod(lambda: iter([])),
    }
    if with_modules:
        ns["modules"] = staticmethod(lambda: iter([]))
    return type(cls_name, (), ns)()


def test_verify_does_not_crash_without_modules(tmp_path):
    snap = _make_snapshot(tmp_path)
    arch = ql.detect_architecture(snap)
    fake = _fake_model(with_modules=False)
    chk = ql.verify_loaded_model(fake, snap, arch)
    assert chk["ok"] is False
    assert any("modules()" in p for p in chk["problems"])


def test_verify_rejects_a_silent_model_swap(tmp_path):
    snap = _make_snapshot(tmp_path)
    arch = ql.detect_architecture(snap)
    fake = _fake_model(cls_name="SomeoneElsesModel")
    chk = ql.verify_loaded_model(fake, snap, arch)
    assert chk["ok"] is False
    assert any("silent model swap" in p for p in chk["problems"])


def test_verify_detects_hidden_layer_and_vocab_mismatch(tmp_path):
    snap = _make_snapshot(tmp_path)
    arch = ql.detect_architecture(snap)
    chk = ql.verify_loaded_model(_fake_model(n_layers=12), snap, arch)
    assert chk["ok"] is False
    assert any("12" in p and "24" in p for p in chk["problems"])
    chk2 = ql.verify_loaded_model(_fake_model(vocab=1000), snap, arch)
    assert chk2["ok"] is False
    assert any("vocab" in p or "embedding rows" in p for p in chk2["problems"])


def test_verify_accepts_a_matching_module_tree(tmp_path):
    snap = _make_snapshot(tmp_path)
    arch = ql.detect_architecture(snap)
    chk = ql.verify_loaded_model(_fake_model(), snap, arch)
    assert chk["ok"] is True, chk["problems"]
    assert chk["identity_verified"] is True
    assert chk["embedding_shape"] == [EXPECTED_VOCAB_SIZE, EXPECTED_HIDDEN_SIZE]
    assert chk["num_hidden_layers"] == 24
    assert chk["tied_embeddings"] is True
    assert chk["n_checkpoint_tensors"] == 488
    assert chk["text_tensors_planned"] == 320


def test_verify_rejects_a_none_model():
    chk = ql.verify_loaded_model(None, "/nonexistent", {})
    assert chk["ok"] is False and chk["problems"]


def test_tokenizer_verification():
    class FakeTok:
        chat_template = None
        eos_token = None
        pad_token = None

        def __len__(self):
            return 999999

        def __call__(self, text, **kw):
            return {"input_ids": [1, 2, 3]}

        def decode(self, ids, **kw):
            return "x"

    r = ql.verify_tokenizer(FakeTok(), {"text_vocab_size": 248320})
    assert r["ok"] is False
    assert any("vocab" in p for p in r["problems"])
    assert any("chat template" in p for p in r["problems"])


# ================================ C. cross-transaction ledger ownership
def test_two_transactions_same_tensor_revert_n2_updates_each_owner(backend, cfg, tmp_project):
    """The reported bug: both reverted IDs were dumped on the latest transaction."""
    rt = _runtime(backend, cfg, tmp_project)
    pristine = backend.get_tensor(N1)

    rt.begin_transaction()
    a = rt.call("apply_edit", {"operation": _op(N1, [0]), "reason": "A"})
    rt.commit_transaction()
    rt.begin_transaction()
    b = rt.call("apply_edit", {"operation": _op(N1, [1], 0.12), "reason": "B"})
    rt.commit_transaction()
    assert a["txn_id"] != b["txn_id"]

    rev = rt.call("revert", {"n": 2})
    assert rev["ok"] is True and rev["verified"] is True
    assert rev["undone"] == [b["edit_id"], a["edit_id"]]     # newest first
    assert np.array_equal(backend.get_tensor(N1), pristine), "tensor fully restored"
    assert rt.applied == []

    idx = json.load(open(rt.ledger.index_file))
    # ownership: each ID is recorded only under its own transaction
    assert set(idx[a["txn_id"]]["reverted_edit_ids"]) == {a["edit_id"]}
    assert set(idx[b["txn_id"]]["reverted_edit_ids"]) == {b["edit_id"]}
    assert idx[a["txn_id"]]["status"] == "reverted"
    assert idx[b["txn_id"]]["status"] == "reverted"
    for t in idx.values():
        assert set(t["reverted_edit_ids"]) <= set(t["edit_ids"])
    # the durable per-transaction ledgers agree
    la = json.load(open(Path(rt.ledger.root) / a["txn_id"] / "ledger.json"))
    lb = json.load(open(Path(rt.ledger.root) / b["txn_id"] / "ledger.json"))
    assert {(e["edit_id"], e["status"]) for e in la["edits"]} == {(a["edit_id"], "reverted")}
    assert {(e["edit_id"], e["status"]) for e in lb["edits"]} == {(b["edit_id"], "reverted")}
    # transaction A is NOT still marked applied
    assert idx[a["txn_id"]]["outcome"] == "committed"
    assert idx[a["txn_id"]]["edits_reverted"] == 1


def test_three_transactions_partial_revert_keeps_correct_owners(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    ids = []
    for i in range(3):
        rt.begin_transaction()
        a = rt.call("apply_edit", {"operation": _op(N1, [i], 0.05 + i * 0.01),
                                   "reason": f"t{i}"})
        rt.commit_transaction()
        ids.append((a["edit_id"], a["txn_id"]))
    rev = rt.call("revert", {"n": 2})
    assert rev["ok"] and rev["undone"] == [ids[2][0], ids[1][0]]
    idx = json.load(open(rt.ledger.index_file))
    reverted = {eid for t in idx.values() for eid in t["reverted_edit_ids"]}
    assert reverted == {ids[2][0], ids[1][0]}
    assert ids[0][0] not in reverted
    assert set(idx[ids[0][1]]["reverted_edit_ids"]) == set()
    assert idx[ids[0][1]]["status"] == "committed"


def test_ledger_mark_edits_reverted_rejects_foreign_ids(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    rt.begin_transaction()
    a = rt.call("apply_edit", {"operation": _op(N1, [0]), "reason": "A"})
    tx_a = rt.txn.txn_id
    rt.commit_transaction()
    rt.begin_transaction()
    b = rt.call("apply_edit", {"operation": _op(N2, [0]), "reason": "B"})
    tx_b = rt.txn.txn_id
    rt.commit_transaction()
    with pytest.raises(Exception, match="different transaction"):
        rt.ledger.mark_edits_reverted(tx_a, [b["edit_id"]])
    assert rt.ledger.index[tx_a]["reverted_edit_ids"] == []


# ==================== D. committed revert, dependencies, atomicity, restart
def test_revert_committed_transaction_restores_state(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    pristine = backend.get_tensor(N1)
    rt.begin_transaction()
    a1 = rt.call("apply_edit", {"operation": _op(N1, [0]), "reason": "1"})
    tx = rt.txn.txn_id
    rt.call("apply_edit", {"operation": _op(N1, [2], 0.07), "reason": "2"})
    rt.commit_transaction()
    assert rt.applied
    rev = rt.call("revert_transaction", {"txn_id": tx})
    assert rev["ok"] is True and rev["verified"] is True
    assert sorted(rev["undone"]) == sorted([a1["edit_id"], rt.ledger.index[tx]["edit_ids"][1]])
    assert np.array_equal(backend.get_tensor(N1), pristine)
    assert rt.applied == []
    idx = rt.ledger.index[tx]
    assert idx["status"] == "reverted"
    assert set(idx["reverted_edit_ids"]) == set(idx["edit_ids"])


def test_revert_committed_transaction_is_idempotent(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    rt.begin_transaction()
    rt.call("apply_edit", {"operation": _op(N1, [0]), "reason": "1"})
    tx = rt.txn.txn_id
    rt.commit_transaction()
    r1 = rt.call("revert_transaction", {"txn_id": tx})
    r2 = rt.call("revert_transaction", {"txn_id": tx})
    assert r1["ok"] and r1["undone"]
    assert r2["ok"] and r2["undone"] == [] and r2["verified"] is True


def test_revert_transaction_unknown_id_is_a_clear_error(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    rev = rt.call("revert_transaction", {"txn_id": "txn_does_not_exist"})
    assert rev["ok"] is False
    assert "unknown transaction" in rev["error"]
    assert rev["undone"] == []


def test_revert_older_transaction_is_refused_while_newer_is_applied(backend, cfg, tmp_project):
    """Reverting A alone would destroy B's pre-image: refuse, change nothing."""
    rt = _runtime(backend, cfg, tmp_project)
    rt.begin_transaction()
    a = rt.call("apply_edit", {"operation": _op(N1, [0]), "reason": "A"})
    tx_a = rt.txn.txn_id
    rt.commit_transaction()
    rt.begin_transaction()
    b = rt.call("apply_edit", {"operation": _op(N1, [1], 0.12), "reason": "B"})
    tx_b = rt.txn.txn_id
    rt.commit_transaction()
    after_b = backend.tensor_hash(N1)
    ledger_before = json.load(open(rt.ledger.index_file))

    rev = rt.call("revert_transaction", {"txn_id": tx_a})
    assert rev["ok"] is False
    assert rev["state_changed"] is False
    assert [d["txn_id"] for d in rev["blocked_by"]] == [tx_b]
    assert backend.tensor_hash(N1) == after_b, "no tensor was modified"
    assert json.load(open(rt.ledger.index_file)) == ledger_before
    assert {x["edit_id"] for x in rt.applied} == {a["edit_id"], b["edit_id"]}

    # cascade reverts B then A, newest-first -> pristine
    pristine_hash = rt.ledger.index[tx_a]["tensors"]
    casc = rt.call("revert_transaction", {"txn_id": tx_a, "cascade": True})
    assert casc["ok"] is True, casc
    assert casc["cascaded"] == [tx_b]
    assert rt.applied == []
    assert pristine_hash == [N1]
    assert rt.ledger.index[tx_a]["status"] == "reverted"
    assert rt.ledger.index[tx_b]["status"] == "reverted"


def test_cascade_leaves_tensor_pristine(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    pristine = backend.get_tensor(N1)
    rt.begin_transaction()
    rt.call("apply_edit", {"operation": _op(N1, [0]), "reason": "A"})
    tx_a = rt.txn.txn_id
    rt.commit_transaction()
    rt.begin_transaction()
    rt.call("apply_edit", {"operation": _op(N1, [1], 0.2), "reason": "B"})
    rt.commit_transaction()
    rev = rt.call("revert_transaction", {"txn_id": tx_a, "cascade": True})
    assert rev["ok"] is True
    assert np.array_equal(backend.get_tensor(N1), pristine)


def test_rollback_failure_is_atomic_and_leaves_ledger_unchanged(backend, cfg, tmp_project,
                                                               monkeypatch):
    """First restore write succeeds, second explodes: the first must be recovered."""
    rt = _runtime(backend, cfg, tmp_project)
    rt.begin_transaction()
    rt.call("apply_edit", {"operation": _op(N1, [0]), "reason": "1"})
    rt.call("apply_edit", {"operation": _op(N2, [0]), "reason": "2"})
    tx = rt.txn.txn_id
    rt.commit_transaction()
    before = {N1: backend.tensor_hash(N1), N2: backend.tensor_hash(N2)}
    ledger_before = json.load(open(rt.ledger.index_file))

    real_set = backend.set_tensor
    state = {"failed": False}

    # restore order is newest-edit first, so N1 is written second: the first
    # write (N2) succeeds and must be rolled back when N1 explodes
    def flaky(name, value):
        if name == N1 and not state["failed"]:
            state["failed"] = True
            raise RuntimeError("device lost during restore")
        return real_set(name, value)

    monkeypatch.setattr(backend, "set_tensor", flaky)
    rev = rt.call("revert_transaction", {"txn_id": tx})
    assert rev["ok"] is False
    assert rev["recovered"] is True
    assert "device lost" in rev["error"]
    # the successful first write was rolled back to the pre-rollback value
    assert {n: backend.tensor_hash(n) for n in (N1, N2)} == before
    assert json.load(open(rt.ledger.index_file)) == ledger_before
    assert rt.ledger.index[tx]["reverted_edit_ids"] == []
    assert {x["edit_id"] for x in rt.applied} == set(rt.ledger.index[tx]["edit_ids"])
    monkeypatch.undo()
    # once the device works again the revert succeeds exactly
    rev2 = rt.call("revert_transaction", {"txn_id": tx})
    assert rev2["ok"] is True, rev2


def test_missing_journal_refuses_without_touching_state(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    rt.begin_transaction()
    rt.call("apply_edit", {"operation": _op(N1, [0]), "reason": "1"})
    tx = rt.txn.txn_id
    rt.commit_transaction()
    h = backend.tensor_hash(N1)
    for f in (Path(rt.ledger.root) / tx).glob("*.npy"):
        f.unlink()
    rev = rt.call("revert_transaction", {"txn_id": tx})
    assert rev["ok"] is False
    assert rev["state_changed"] is False
    assert backend.tensor_hash(N1) == h
    assert rt.ledger.index[tx]["reverted_edit_ids"] == []


def test_restart_recovery_reverts_a_committed_transaction(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    pristine = backend.get_tensor(N1)
    rt.begin_transaction()
    a = rt.call("apply_edit", {"operation": _op(N1, [0]), "reason": "before restart"})
    tx = rt.txn.txn_id
    rt.commit_transaction()
    kept = backend.tensor_hash(N1)

    # "restart": a new runtime on the same durable ledger, same tensors
    rt2 = _runtime(backend, cfg, tmp_project)
    assert tx in rt2.recovery["recovered_transactions"]
    assert [x["edit_id"] for x in rt2.applied] == [a["edit_id"]]
    assert rt2.applied[0]["txn_id"] == tx
    rev = rt2.call("revert_transaction", {"txn_id": tx})
    assert rev["ok"] is True and rev["verified"] is True
    assert backend.tensor_hash(N1) != kept
    assert np.array_equal(backend.get_tensor(N1), pristine)
    assert rt2.applied == []


def test_restart_allows_per_edit_revert_of_first_edit_on_tensor(backend, cfg, tmp_project):
    rt = _runtime(backend, cfg, tmp_project)
    pristine = backend.get_tensor(N1)
    rt.begin_transaction()
    rt.call("apply_edit", {"operation": _op(N1, [0]), "reason": "first"})
    rt.commit_transaction()
    rt2 = _runtime(backend, cfg, tmp_project)
    rev = rt2.call("revert", {"n": 1})
    assert rev["ok"] is True and rev["verified"] is True
    assert np.array_equal(backend.get_tensor(N1), pristine)


# ============================================== E. NVIDIA agent synchronization
def _reply(tool, args):
    """A model reply carrying exactly one fenced JSON tool call."""
    return "```json\n" + json.dumps({"tool": tool, "args": args}) + "\n```"


class _FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.i = 0

    def configured(self):
        return True

    def chat(self, *a, **k):
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return r


def _agent(backend, cfg, tmp_project, replies):
    from aladdin.agent.nvidia_agent import NvidiaAgent
    rt = _runtime(backend, cfg, tmp_project)
    cfg = json.loads(json.dumps(cfg))
    cfg["agent"] = dict(cfg["agent"], max_turns=6, max_steps=6)
    ag = NvidiaAgent(cfg, rt, Path(tmp_project["paths"]["experiments"]))
    ag.client = _FakeClient(replies)
    return ag, rt


def test_agent_tracks_applied_and_reverted_edits(backend, cfg, tmp_project):
    ag, rt = _agent(backend, cfg, tmp_project, [
        _reply("apply_edit", {"operation": _op(N1, [0], 0.05), "reason": "probe"}),
        _reply("revert", {"n": 1}),
        _reply("finish", {"summary": "done"}),
    ])
    out = ag.run_step(0, [], 0.5, {}, 10)
    assert out["ok"] is True
    assert out["applied"] == [], "a reverted edit must not be reported active"
    assert rt.applied == []
    log = [json.loads(l) for l in ag.transcript.read_text().splitlines() if l.strip()]
    assert any(e["event"] == "revert_applied" and e["undone"] for e in log)
    assert not any(e["event"] == "revert_failed" for e in log)


def test_agent_failed_revert_keeps_edit_active(backend, cfg, tmp_project, monkeypatch):
    ag, rt = _agent(backend, cfg, tmp_project, [
        _reply("apply_edit", {"operation": _op(N1, [0], 0.05), "reason": "probe"}),
        _reply("revert", {"n": 1}),
        _reply("finish", {"summary": "done"}),
    ])

    # the edit must be applied first, so let the first write through
    real = backend.set_tensor
    calls = {"n": 0}

    def setter(name, value):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("restore exploded")
        return real(name, value)
    monkeypatch.setattr(backend, "set_tensor", setter)
    out = ag.run_step(0, [], 0.5, {}, 10)
    assert out["ok"] is True
    assert len(out["applied"]) == 1, "failed revert must keep the edit active"
    log = [json.loads(l) for l in ag.transcript.read_text().splitlines() if l.strip()]
    assert any(e["event"] == "revert_failed" for e in log)


def test_agent_revert_transaction_by_id(backend, cfg, tmp_project):
    ag, rt = _agent(backend, cfg, tmp_project, [])
    rt.begin_transaction()
    a = rt.call("apply_edit", {"operation": _op(N1, [0]), "reason": "1"})
    tx = rt.txn.txn_id
    rt.commit_transaction()
    ag.client = _FakeClient([
        _reply("revert_transaction", {"txn_id": tx}),
        _reply("finish", {"summary": "reverted txn"}),
    ])
    out = ag.run_step(0, [], 0.5, {}, 10)
    assert out["ok"] is True and out["applied"] == []
    assert rt.applied == []
    log = [json.loads(l) for l in ag.transcript.read_text().splitlines() if l.strip()]
    rev = [e for e in log if e.get("event") == "revert_applied"][-1]
    assert rev["txn_id"] == tx and rev["undone"] == [a["edit_id"]]
    assert rev["runtime_active"] == []


# ====================================== F. Hindi / Hinglish evaluation coverage
REQUIRED_EXAMPLES = [
    "mujhe ye samjha", "muje ye samja", "iska kya matlab hai",
    "ab isko thik karo", "kal wala kaam continue karo",
    "mere liye ek accha assistant banao", "मुझे यह समझाओ",
    "इसका मतलब क्या है", "अब इसे ठीक करो",
    "मुझे एक अच्छा AI assistant बनाना है",
]


def _shipped_splits():
    from aladdin.evaluation.datasets import Dataset
    root = Path(__file__).resolve().parent.parent / "datasets"
    dev = Dataset.load(root / "dev.jsonl", "dev")
    held = Dataset.load(root / "heldout.jsonl", "heldout")
    return dev, held


def test_required_hinglish_examples_are_in_the_evaluation_sets():
    dev, held = _shipped_splits()
    prompts = {r["prompt"] for r in dev.rows} | {r["prompt"] for r in held.rows}
    for p in REQUIRED_EXAMPLES:
        assert p in prompts, f"required example missing from evaluation sets: {p}"
    # spelling variants of the same intent are represented on both splits
    assert "mujhe ye samjha" in {r["prompt"] for r in dev.rows}
    assert "muje ye samja" in {r["prompt"] for r in held.rows}


def test_evaluation_sets_separate_dev_and_heldout():
    dev, held = _shipped_splits()
    assert not (dev.ids() & held.ids())
    assert not ({r["prompt"] for r in dev.rows} & {r["prompt"] for r in held.rows})
    assert len(dev.by_category()) == 10 and len(held.by_category()) == 10
    # every aspect from the spec is represented by at least one category
    cats = set(dev.by_category()) | set(held.by_category())
    for needed in ("roman_hinglish", "devanagari_hindi", "mixed_code_switch",
                   "context_followup", "instruction_following", "natural_response",
                   "spelling_robustness", "question_answering", "factuality"):
        assert needed in cats


def test_heldout_is_not_used_by_the_search_loop():
    from aladdin.evaluation.datasets import assert_no_heldout_leak
    dev, held = _shipped_splits()
    assert_no_heldout_leak(dev.ids(), held)
    with pytest.raises(AssertionError):
        assert_no_heldout_leak(dev.ids() | {next(iter(held.ids()))}, held)


# ================================================== G. WSP contract (direct edits)
def test_direct_edit_sparse_delta_checkpoint_and_exact_rollback(backend, cfg, tmp_project):
    from aladdin.checkpoints.manager import CheckpointManager
    store = DeltaStore(tmp_project["paths"]["deltas"])
    ed = TensorEditor(backend, cfg, store)
    ck = CheckpointManager(backend, tmp_project["paths"]["checkpoints"],
                           deltas_root=tmp_project["paths"]["deltas"],
                           backend_factory=lambda: type(backend)(cfg))
    ck.save_base()
    h0 = backend.tensor_hash(N1)
    res = ed.apply(_op(N1, [0, 3], 0.1), reason="direct WSP edit")
    assert res["verified"] and res["changed_entries"] > 0
    man, arrays = store.load(res["edit_id"])
    assert man["tensors"][0]["dtype"] == "float32"
    assert arrays["t0_idx"].size == res["changed_entries"]
    ck.save("after_edit", txn="txn_test")
    r = ck.restore("after_edit")
    assert r["ok"] and backend.tensor_hash(N1) == res["edited_sha256"]
    rb = ck.rebuild_chain_after_reset(backend, [])
    assert rb["ok"] and backend.tensor_hash(N1) == h0
