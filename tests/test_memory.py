"""Phase (d): vector memory — validated-only writes, staleness guard, reuse path.

Uses a deterministic fake embedder so CI never downloads embedding models.
"""
import hashlib
import uuid

from analyst.data import build_manifest
from analyst.eval.data_gen import generate
from analyst.graph.control_loop import build_graph
from analyst.graph.state import (
    Artifact, Budgets, ExecutionOutcome, TaskSpec, ValidatedClaim,
)
from analyst.llm.replay import ReplayClient
from analyst.memory.store import MemoryStore
from analyst.sandbox.runner import make_runner


class FakeEmbed:
    """Deterministic bag-of-characters embedding. Terrible semantics, perfect tests:
    identical strings are identical vectors, and similar strings are close."""

    def __call__(self, input):
        out = []
        for text in input:
            v = [0.0] * 32
            for tok in text.lower().split():
                v[hash(tok) % 32] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out

    def embed_query(self, input):
        return self(input)  # chroma >=1.5 uses a separate query-embedding hook

    @staticmethod
    def name():
        return "fake-bag-of-words"

    def is_legacy(self):
        return True  # skip config round-trip machinery for this test double


def make_store(tmp_path):
    return MemoryStore(path=tmp_path / "chroma", embedding_function=FakeEmbed())


def stored_run(store, dataset_hash, goal="total downtime hours per machine"):
    dag = {"t1": TaskSpec(id="t1", goal=goal, status="done")}
    artifacts = {
        "t1": Artifact(
            task_id="t1", code="...",
            outcome=ExecutionOutcome(status="ok", result={"answer": "M13 tops", "data": {"M13": 196.54}}),
        )
    }
    claims = [ValidatedClaim(task_id="t1", claim="M13 has most downtime", verdict="pass")]
    return store.store_run("run-1", "which machine has most downtime", dataset_hash,
                           dag, artifacts, claims)


def test_only_validated_results_are_stored(tmp_path):
    store = make_store(tmp_path)
    dag = {"t1": TaskSpec(id="t1", goal="g", status="done")}
    artifacts = {"t1": Artifact(task_id="t1", code="",
                                outcome=ExecutionOutcome(status="ok", result={"x": 1}))}
    for verdict in ("caveat", "fail"):
        n = store.store_run("r", "q", "hash-a", dag, artifacts,
                            [ValidatedClaim(task_id="t1", claim="c", verdict=verdict)])
        assert n == 0  # a memory of caveats/mistakes is worse than no memory
    n = store.store_run("r", "q", "hash-a", dag, artifacts,
                        [ValidatedClaim(task_id="t1", claim="c", verdict="pass")])
    assert n == 1


def test_roundtrip_and_staleness_guard(tmp_path):
    store = make_store(tmp_path)
    assert stored_run(store, "hash-a") == 1

    hits = store.query("which machine has the most downtime hours", "hash-a")
    assert len(hits) == 1 and hits[0].result["data"]["M13"] == 196.54

    # same question, different data bytes -> the hash filter must hide the memory
    assert store.query("which machine has the most downtime hours", "hash-B") == []


def test_upsert_replaces_same_goal_same_data(tmp_path):
    store = make_store(tmp_path)
    stored_run(store, "hash-a")
    stored_run(store, "hash-a")  # identical goal+hash -> upsert, not duplicate
    assert store.col.count() == 1


def test_full_graph_reuse_path(tmp_path, monkeypatch):
    """Run 2 of the same question plans a reuse task: no executor call, no sandbox,
    artifact carries provenance."""
    monkeypatch.setenv("ANALYST_SANDBOX", "subprocess")
    data_dir = tmp_path / "sample"
    generate(data_dir, seed=7)
    manifest = build_manifest(data_dir)
    store = make_store(tmp_path)
    stored_run(store, manifest.content_hash)

    mem_id = hashlib.sha256(
        f"{manifest.content_hash}|total downtime hours per machine".encode()
    ).hexdigest()[:16]
    plan_reuse = (
        '{"tasks": [{"id": "t1", "goal": "total downtime hours per machine", '
        f'"kind": "reuse", "reuse_of": "{mem_id}", "depends_on": []}}]}}'
    )
    script = {"planner": [{"text": plan_reuse}],
              "validator": [{"text": '{"claims": [{"task_id": "t1", "claim": "reused", "verdict": "pass", "note": ""}]}'}]}
    # note: NO executor entries — if the reuse path ever called the executor,
    # this test would fail with CassetteMiss.
    graph = build_graph(
        ReplayClient(cassette_dir=None, script=script),
        make_runner("subprocess"),
        critic_enabled=True,  # reuse must bypass the critic too
        validator_enabled=True,
        memory=store,
    )
    state = graph.invoke(
        {"run_id": uuid.uuid4().hex[:8], "question": "which machine has most downtime",
         "dataset": manifest, "budgets": Budgets()},
        config={"recursion_limit": 60},
    )
    t1 = state["dag"]["t1"]
    assert t1.status == "done"
    art = state["artifacts"]["t1"]
    assert art.model == "memory"                 # provenance
    assert "run-1" in art.code                   # cites the source run
    assert art.outcome.result["data"]["M13"] == 196.54


def test_hallucinated_memory_id_downgrades_to_recompute(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYST_SANDBOX", "subprocess")
    data_dir = tmp_path / "sample"
    generate(data_dir, seed=7)
    manifest = build_manifest(data_dir)

    plan_bad_reuse = (
        '{"tasks": [{"id": "t1", "goal": "count work orders", '
        '"kind": "reuse", "reuse_of": "deadbeef00000000", "depends_on": []}]}'
    )
    code = ('```python\nimport os, json\nimport pandas as pd\n'
            'df = pd.read_csv(os.path.join(os.environ["DATA_DIR"], "work_orders.csv"))\n'
            'json.dump({"answer": f"{len(df)} work orders", "data": {"n": len(df)}},'
            ' open("result.json", "w"))\n```')
    script = {"planner": [{"text": plan_bad_reuse}], "executor": [{"text": code}]}
    graph = build_graph(
        ReplayClient(cassette_dir=None, script=script),
        make_runner("subprocess"),
        critic_enabled=False, validator_enabled=False,
        memory=make_store(tmp_path),
    )
    state = graph.invoke(
        {"run_id": "r", "question": "q", "dataset": manifest, "budgets": Budgets()},
        config={"recursion_limit": 60},
    )
    t1 = state["dag"]["t1"]
    assert t1.kind == "python" and t1.reuse_of is None  # downgraded, then computed
    assert t1.status == "done"
