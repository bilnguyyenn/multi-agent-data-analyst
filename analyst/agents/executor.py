"""Executor agent: one sub-task -> code -> sandboxed run -> Artifact.

The executor is MECHANICAL-tier by design: by the time it runs, the hard thinking
(decomposition) already happened. Its job is 'execute a well-specified instruction'.
It gets its dependencies' results inlined, and on retry it gets the exact failure."""
from __future__ import annotations

import json
from pathlib import Path

from analyst.agents import prompts
from analyst.graph.state import Artifact, DatasetManifest, ExecutionOutcome, TaskSpec
from analyst.llm.client import LLMClient
from analyst.llm.parsing import ModelOutputError, extract_code
from analyst.sandbox.runner import SandboxRunner

DEP_RESULT_MAX_CHARS = 1500  # inlined dependency results are summaries, not dumps

# Injected ahead of every generated script. This is an ENVIRONMENT GUARANTEE
# (like a Jupyter kernel with preloaded libs), not model babysitting: it makes the
# entire missing-import failure class structurally impossible instead of hoping a
# 3B model follows instructions. Re-imports in generated code are harmless.
# Motivated by a live run where the model wrote pandas code without `import pandas`
# three times in a row, ignoring the traceback each retry.
CODE_PREAMBLE = (
    "import os, json\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import duckdb\n"
    "# Compat shims for commonly-hallucinated API names (each documents a real\n"
    "# live failure a small model repeated across retries):\n"
    "try:\n"
    "    duckdb.DuckDBPyRelation.fetch_df = duckdb.DuckDBPyRelation.fetchdf\n"
    "except AttributeError:\n"
    "    pass\n"
    "# --- end of environment preamble; generated code follows ---\n"
)


def run_task(
    llm: LLMClient,
    sandbox: SandboxRunner,
    task: TaskSpec,
    dataset: DatasetManifest,
    dep_artifacts: dict[str, Artifact],
    timeout_s: int = 60,
) -> Artifact:
    deps_section = ""
    if dep_artifacts:
        lines = []
        for tid, art in dep_artifacts.items():
            payload = json.dumps(art.outcome.result)[:DEP_RESULT_MAX_CHARS]
            lines.append(f"- {tid}: {payload}")
        deps_section = prompts.EXECUTOR_DEPS.format(deps="\n".join(lines))

    feedback_section = (
        prompts.EXECUTOR_FEEDBACK.format(feedback=task.feedback) if task.feedback else ""
    )

    resp = llm.complete(
        role="executor",
        system=prompts.EXECUTOR_SYSTEM,
        user=prompts.EXECUTOR_USER.format(
            goal=task.goal,
            schema=dataset.schema_text(),
            deps_section=deps_section,
            feedback_section=feedback_section,
        ),
    )

    try:
        code = CODE_PREAMBLE + extract_code(resp.text)
    except ModelOutputError as e:
        # Unparseable output is a normal, retryable failure — same taxonomy as a crash.
        outcome = ExecutionOutcome(status="error", stderr=f"[parse] {e}")
        code = resp.text
    else:
        outcome = sandbox.run(
            code,
            Path(dataset.root),
            timeout_s=timeout_s,
            deps={tid: art.outcome.result for tid, art in dep_artifacts.items()},
        )

    return Artifact(
        task_id=task.id,
        code=code,
        outcome=outcome,
        model=resp.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )
