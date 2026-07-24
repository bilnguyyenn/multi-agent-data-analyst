"""Robust extraction of JSON / code from model output.

Small local models decorate answers with prose despite instructions. We never
`json.loads(raw_text)` directly: we find the first balanced JSON object, and we
prefer fenced code blocks but fall back to raw text. Every parse failure raises
with the offending text attached — parse errors are retryable executor failures,
not crashes.
"""
from __future__ import annotations

import json
import re
from typing import Any


class ModelOutputError(ValueError):
    pass


def extract_json(text: str) -> Any:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
            elif c == '"' and not esc:
                in_str = not in_str
            elif not in_str:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break  # malformed; try next '{'
        start = text.find("{", start + 1)
    raise ModelOutputError(f"No parseable JSON object in model output:\n{text[:500]}")


_FENCE = re.compile(r"```(?:python|py|sql)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = _FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()  # longest block = the program
    if "import " in text or "result.json" in text:
        return text.strip()  # model skipped the fence but wrote code
    raise ModelOutputError(f"No code block in model output:\n{text[:500]}")
