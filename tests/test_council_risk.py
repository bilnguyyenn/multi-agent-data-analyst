from analyst.graph.risk import assess_council_risk
from analyst.graph.state import Artifact, ExecutionOutcome, TaskSpec


def artifact(code: str) -> dict[str, Artifact]:
    return {
        "t1": Artifact(
            task_id="t1",
            code=code,
            outcome=ExecutionOutcome(status="ok", result={"answer": "ok"}),
        )
    }


def test_adaptive_router_skips_simple_lookup():
    decision = assess_council_risk(
        round_number=0,
        mode="adaptive",
        question="How many work orders are in the dataset?",
        dag={"t1": TaskSpec(id="t1", goal="count rows", status="done")},
        artifacts=artifact("df = pd.read_csv('work_orders.csv')"),
    )

    assert decision.invoked is False
    assert decision.risk_score < decision.threshold


def test_filesystem_path_join_is_not_an_analysis_join():
    decision = assess_council_risk(
        round_number=0,
        mode="adaptive",
        question="How many work orders are in the dataset?",
        dag={"t1": TaskSpec(id="t1", goal="count rows", status="done")},
        artifacts=artifact(
            "df = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'work_orders.csv'))"
        ),
    )

    assert decision.invoked is False
    assert "executed analysis joins tables" not in decision.reasons


def test_adaptive_router_invokes_for_statistical_or_temporal_claim():
    decision = assess_council_risk(
        round_number=0,
        mode="adaptive",
        question="Has downtime improved since July, and is the change meaningful?",
        dag={"t1": TaskSpec(id="t1", goal="compare periods", status="done")},
        artifacts=artifact("df = pd.read_csv('downtime.csv')"),
    )

    assert decision.invoked is True
    assert "temporal or causal comparison" in decision.reasons
    assert "statistical or uncertainty-sensitive claim" in decision.reasons


def test_joined_rate_crosses_risk_threshold():
    decision = assess_council_risk(
        round_number=0,
        mode="adaptive",
        question="What fraction of total production was defective?",
        dag={"t1": TaskSpec(id="t1", goal="compute defective fraction", status="done")},
        artifacts=artifact(
            "a = pd.read_csv('orders.csv')\n"
            "b = pd.read_csv('defects.csv')\n"
            "result = a.merge(b, on='order_id')"
        ),
    )

    assert decision.invoked is True
    assert "join plus rate creates fan-out/denominator risk" in decision.reasons


def test_forced_modes_override_risk_score():
    kwargs = {
        "round_number": 0,
        "question": "How many rows?",
        "dag": {"t1": TaskSpec(id="t1", goal="count", status="done")},
        "artifacts": artifact("df = pd.read_csv('x.csv')"),
    }
    assert assess_council_risk(mode="always", **kwargs).invoked is True
    assert assess_council_risk(mode="off", **kwargs).invoked is False
