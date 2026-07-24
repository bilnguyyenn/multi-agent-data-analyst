"""The environment preamble makes missing-import failures structurally impossible.
Regression test for a live 3B run that wrote pandas code with no imports, 3x."""
import pytest

from analyst.agents.executor import run_task
from analyst.data import build_manifest
from analyst.eval.data_gen import generate
from analyst.graph.state import TaskSpec
from analyst.llm.client import LLMResponse
from analyst.sandbox.runner import make_runner

# What the 3B actually did: pandas + json usage, zero imports.
IMPORTLESS_CODE = '''```python
df = pd.read_csv(os.path.join(os.environ["DATA_DIR"], "downtime_events.csv"))
per = df.groupby("machine_id")["downtime_hours"].sum()
json.dump({"answer": f"top is {per.idxmax()}", "data": {"machine": per.idxmax()}},
          open("result.json", "w"))
```'''


class FakeLLM:
    def complete(self, *, role, system, user):
        return LLMResponse(text=IMPORTLESS_CODE, model="fake-3b")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("data") / "sample"
    generate(d, seed=7)
    return build_manifest(d)


def test_importless_code_still_runs(dataset, monkeypatch):
    monkeypatch.setenv("ANALYST_SANDBOX", "subprocess")
    sandbox = make_runner("subprocess")
    task = TaskSpec(id="t1", goal="top downtime machine")
    art = run_task(FakeLLM(), sandbox, task, dataset, {})
    assert art.outcome.status == "ok", art.outcome.stderr
    assert art.outcome.result["data"]["machine"].startswith("M")
    assert art.code.startswith("import os, json")  # preamble is present in the artifact
