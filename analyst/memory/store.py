"""Vector memory: 'have I answered something like this before?' (phase d).

Design rules (ARCHITECTURE.md §2.5), each defensible on its own:

1. WRITE ONLY VALIDATED RESULTS. A sub-result enters memory only if its claim
   passed the statistical validator. Rejected/caveated work never becomes
   reusable "truth" — a memory of mistakes is worse than no memory.
2. STALENESS GUARD = DATASET CONTENT HASH. Queries filter on the exact hash of
   the data files. A cached answer from *different* data isn't a cache hit,
   it's a wrong answer with confidence. Change one row -> the whole memory for
   that dataset is invisible (correctly so).
3. CHROMA STORES POINTERS + SMALL PAYLOADS, NOT TRUTH. The system of record is
   the run ledger (Postgres in phase e; run_id/task_id metadata today). Vector
   DBs are for finding things, not for owning them.
4. UPSERT KEYED ON (dataset_hash, goal): re-answering the same goal on the same
   data replaces the entry instead of accumulating near-duplicates.

Embeddings: Chroma's default ONNX MiniLM (~80MB, downloaded once, local, free).
Tests inject a deterministic fake embedder — CI never downloads models.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from analyst.graph.state import Artifact, MemoryHit, TaskSpec, ValidatedClaim

DEFAULT_DIR = ".analyst_memory"
RESULT_MAX_BYTES = 4000


class MemoryStore:
    def __init__(self, path: str | Path | None = None, embedding_function=None):
        import chromadb

        self.client = chromadb.PersistentClient(
            path=str(path or os.environ.get("ANALYST_MEMORY_DIR", DEFAULT_DIR))
        )
        kwargs = {}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self.col = self.client.get_or_create_collection(
            "subresults", metadata={"hnsw:space": "cosine"}, **kwargs
        )

    # -- write path ---------------------------------------------------------

    def store_run(
        self,
        run_id: str,
        question: str,
        dataset_hash: str,
        dag: dict[str, TaskSpec],
        artifacts: dict[str, Artifact],
        claims: list[ValidatedClaim],
    ) -> int:
        """Store sub-results whose claims PASSED validation. Returns count stored."""
        stored = 0
        for c in claims:
            if c.verdict != "pass":
                continue
            t, a = dag.get(c.task_id), artifacts.get(c.task_id)
            if not t or not a or a.outcome.result is None:
                continue
            payload = json.dumps(a.outcome.result)
            if len(payload) > RESULT_MAX_BYTES:
                continue  # memory holds answers, not datasets
            mid = hashlib.sha256(f"{dataset_hash}|{t.goal}".encode()).hexdigest()[:16]
            # The QUESTION leads the document: future lookups query with questions,
            # so question->question similarity is the signal; goal + a slice of the
            # result are context. (Found via test: goal+JSON alone didn't retrieve.)
            self.col.upsert(
                ids=[mid],
                documents=[f"{question}\n{t.goal}\n{payload[:300]}"],
                metadatas=[{
                    "run_id": run_id,
                    "task_id": t.id,
                    "dataset_hash": dataset_hash,
                    "goal": t.goal,
                    "result": payload,
                    "question": question,
                }],
            )
            stored += 1
        return stored

    # -- read path ------------------------------------------------------------

    def query(
        self,
        question: str,
        dataset_hash: str,
        k: int = 3,
        min_similarity: float = 0.35,
    ) -> list[MemoryHit]:
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(
            query_texts=[question],
            n_results=min(k, n),
            where={"dataset_hash": dataset_hash},  # the staleness guard
        )
        hits = []
        for i, mid in enumerate(res["ids"][0]):
            similarity = 1.0 - res["distances"][0][i]  # cosine space
            if similarity < min_similarity:
                continue
            m = res["metadatas"][0][i]
            hits.append(
                MemoryHit(
                    id=mid,
                    goal=m["goal"],
                    result=json.loads(m["result"]),
                    run_id=m["run_id"],
                    task_id=m["task_id"],
                    similarity=round(similarity, 3),
                )
            )
        return hits
