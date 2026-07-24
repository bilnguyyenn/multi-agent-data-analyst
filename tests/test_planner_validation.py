import pytest

from analyst.agents.planner import PlanError, make_plan
from analyst.graph.state import DatasetManifest, TableInfo
from analyst.llm.client import LLMResponse


class FakeLLM:
    def __init__(self, text):
        self.text = text

    def complete(self, *, role, system, user):
        return LLMResponse(text=self.text, model="fake")


MANIFEST = DatasetManifest(
    root="x",
    tables=[TableInfo(name="t", path="t.csv", columns=["a"], dtypes={"a": "int64"}, row_count=1)],
    content_hash="h",
)


def test_valid_plan_parses():
    llm = FakeLLM(
        'Here is the plan: {"tasks": [{"id": "t1", "goal": "count rows", "depends_on": [], '
        '"kind": "sql"}, {"id": "t2", "goal": "compare", "depends_on": ["t1"]}]}'
    )
    dag, usage = make_plan(llm, "q", MANIFEST)
    assert set(dag) == {"t1", "t2"}
    assert dag["t2"].depends_on == ["t1"]
    assert dag["t2"].kind == "python"  # default filled in


def test_cyclic_plan_rejected():
    llm = FakeLLM(
        '{"tasks": [{"id": "a", "goal": "x", "depends_on": ["b"]},'
        '{"id": "b", "goal": "y", "depends_on": ["a"]}]}'
    )
    with pytest.raises(PlanError, match="cyclic"):
        make_plan(llm, "q", MANIFEST)


def test_dangling_dependency_rejected():
    llm = FakeLLM('{"tasks": [{"id": "a", "goal": "x", "depends_on": ["ghost"]}]}')
    with pytest.raises(PlanError, match="unknown task"):
        make_plan(llm, "q", MANIFEST)


def test_prose_only_output_rejected():
    llm = FakeLLM("I think we should analyze the data carefully.")
    with pytest.raises(PlanError):
        make_plan(llm, "q", MANIFEST)
