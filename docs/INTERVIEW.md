# Interview defense — talking points for `analyst`

How to present this project under technical questioning, and honest answers to the
hard questions. Numbers are from the committed scorecard (Cerebras `gpt-oss-120b`
reasoning / `gemma-4-31b` mechanical, model held constant across ablation configs).

---

## 30-second pitch

> "It's a multi-agent system that answers business questions against a dataset. A
> planner breaks the question into a task DAG; sandboxed executors write and run
> SQL/pandas; in High-assurance mode, metric, statistical, and data-quality analysts
> independently inspect the executed evidence in parallel. Before that reasoning,
> a deterministic EDA layer profiles missingness, keys, relationships, distributions,
> and outliers so known data risks are visible to both agents and the user.
> An adjudicator resolves disagreement and can request one targeted recomputation
> before a deterministic statistics layer computes intervals, effect sizes, and
> hypothesis tests; confounded comparisons are standardized across common strata to
> detect Simpson reversals. Predictive requests use governed scikit-learn pipelines
> with leakage-aware temporal/group holdouts, training-only model selection, dummy
> baselines, bootstrap metric intervals, and permutation importance before the
> validator gates the final claims. The interesting part
> isn't that it works — it's that I built an evaluation harness with planted
> statistical traps and ablated each review agent on and off, so I can *prove* with
> numbers which components earn their complexity and which don't. The selected
> council-without-critic policy scores 0.86 against 0.56 for a naive baseline. A
> separate 20-question product-analytics suite now tests statistical
> generalization across independently generated held-out seeds without changing the
> original scorecard; I have not claimed its accuracy before a full live run. The
> ablations let me remove two mechanisms that did not justify their cost."

That last sentence is the whole project. Lead with the evaluation, not the agents —
anyone can wire up agents; measuring them is the differentiator.

The measured nuance matters: the efficient Standard path scores 0.80 at 3,429 mean
tokens, while the High-assurance council scores 0.86 at 17,223. The critic adds no
overall accuracy, and adaptive routing fails to save cost, so both are disabled by
default rather than defended as sunk complexity.

---

## If you only prep three answers, prep these

1. **"Isn't this over-engineered?"** — The honest, measured answer below. This is the
   most likely challenge and the one where a quantified reply wins.
2. **"You're executing LLM-generated code — how is that safe?"** — Security thinking.
3. **"Your critic added zero overall. What did you do?"** — Tests whether you'll
   defend sunk complexity or act on evidence. The answer is: remove it from default.

---

## The hard questions

### Q. How do you know the predictive layer isn't leaking its test data?
**Answer:** "The application owns model fitting rather than trusting generated
training code. Model selection uses training-only CV, with preprocessing inside
each pipeline. Temporal holdout and CV keep entire timestamp cohorts together;
group splits keep entities apart. A regression test changes only holdout labels
and verifies identical candidate CV scores and model selection. The versioned
engine benchmark passes 30 case instances across three synthetic seeds, including
target/identifier leakage, join fanout, no-signal, and temporal-shift controls.
Those are reliability checks of fixed specifications, not evidence that an LLM
always selects the correct features or that synthetic accuracy transfers to production."

### Q. What makes this genuinely multi-agent rather than a chain of prompts?
**What they're testing:** whether agents have independent state, work, and conflict.
**Answer:** "The executor pipeline alone is role-based, so I added an analysis
council at the evidence boundary. Three specialist nodes receive the same immutable
artifacts independently and execute as parallel LangGraph `Send` branches. They
return structured conclusions, evidence task ids, risks, confidence, and a canonical
answer key. Agreement is detected deterministically; disagreement invokes a separate
adjudicator, which can send one existing task back for a bounded recomputation. The
UI preserves the three opinions and dissent. Standard mode skips the council; High
assurance invokes it explicitly. That gives me independent work, communication
through shared state, conflict resolution, and a termination policy—not just
sequential prompt labels."

### Q. Isn't a team of agents over-engineered? Why not one big prompt?
**What they're testing:** whether you add complexity reflexively or deliberately.
**Answer:** "I didn't assume it was worth it—I measured seven configurations. A bare
planner+executor scores 0.56, the efficient validator-only path scores 0.80, and the
council-without-critic policy scores 0.86 with perfect tiers 1–3. That last six-point
gain costs about 5× the tokens, so I expose it as High assurance instead of always
on. The critic and adaptive router failed their hypotheses, so I removed both from
the default. The point is complexity I can measure, price, or delete."

### Q. Your critic added zero overall and used 2.36× the tokens. Why keep it?
**What they're testing:** intellectual honesty; will you defend a weak component?
**Answer:** "I don't keep it in the product path. Critic-bench shows it catches 83%
of real flaws, but it also false-positives on 50% of sound analyses. End to end, it
does not improve the 0.80 standard score. In the council path, removing it improves
the score from 0.84 to 0.86 and cuts mean tokens about 13%. I retained the component
only behind `--critic` so the experiment remains reproducible."

### Q. You're running LLM-generated code. How do you keep that from wrecking things?
**What they're testing:** security instinct, defense-in-depth.
**Answer:** "I treat the sandbox as a real security boundary, not an afterthought.
Every execution runs in a throwaway Docker container: no network, read-only root
filesystem, CPU/memory/PID limits, non-root user, and a wall-clock kill. Generated
SQL runs through DuckDB *inside* that container, so the LLM's code and any real
application database are never on the same side of the wall. I also scoped it
honestly — for a single-user portfolio system, per-execution containers are the right
level. I document Firecracker/microVMs as the upgrade path for multi-tenant
isolation, but I deliberately didn't build that because it'd be over-engineering for
this threat model, and I can explain exactly when you'd need it."

### Q. How do you actually know it works? Walk me through evaluation.
**What they're testing:** rigor; the difference between a demo and an engineered system.
**Answer:** "25 questions over synthetic manufacturing data, in four difficulty tiers:
simple lookups, multi-table joins, statistical claims needing caveats, and
deliberately adversarial 'trap' questions where the obvious answer is wrong — a
shift/job-mix confounder, survivorship via decommissioned machines, a one-to-many
join that double-counts. Because I generate the data, gold answers are computed from
the same bytes the agents see, so scoring is objective. Trap scoring rewards honesty:
1.0 for the correct conclusion, 0.5 for the naive number *with the trap named in a
caveat*, 0 for confidently wrong. Then I run the whole thing under four ablation
configs and persist the raw outputs, which can be re-scored at $0. Model calls can
also be recorded as prompt-exact replay cassettes. The published scorecard rejects incomplete
or errored runs and lists the source file for each row. CI tests the orchestration
deterministically with scripted/replayed model responses; I keep the live benchmark
as an explicit evaluation job rather than pretending mocked model calls are a new
live result."

### Q. Why route different models to different agents?
**What they're testing:** cost/latency reasoning, not just "use the biggest model."
**Answer:** "Model choice is config, not code — a table mapping agent role → tier →
concrete model. Reasoning-heavy roles (planner, critic, validator) get the stronger
model; mechanical roles (executor writing well-specified code, summarizing) get a
cheaper one. The eval treats that routing table as an experimental variable, so I can
answer 'how small can the executor model be?' with data rather than a guess. One live
finding I kept: an 8B executor hallucinated column and table names on multi-table
codegen, so the reliable floor sat higher than I expected — that's the kind of thing
you only learn by measuring."

### Q. How does state flow between agents? How do parallel agents not clobber each other?
**What they're testing:** understanding of your own orchestration, not framework buzzwords.
**Answer:** "LangGraph graphs compile statically, but my planner invents a *new* DAG
per question — so the analysis DAG can't be the graph itself. It lives in state as
data, and a deterministic scheduler fans ready tasks out to parallel executors using
LangGraph's `Send`. The concurrency-safety trick is that parallel writers only touch
disjoint keys in state, merged through per-field reducers — so there are no locks and
no lost updates. Every loop is separately budgeted: crash retries, critic revisions,
replans, validation passes. When a budget is exhausted, the objection ships as a
caveat rather than being silently dropped — the system degrades to honesty."

### Q. Your memory layer is a vector DB. How do you stop it serving stale or wrong results?
**What they're testing:** whether "add a vector DB" was cargo-culted.
**Answer:** "Two guardrails by construction. First, only validator-passed results are
ever written — the memory can't cache a bad answer. Second, retrieval filters on the
dataset's content hash, so if the underlying data changed, a semantically-similar past
question simply doesn't match; stale data produces a miss by design, not by luck. And
a hallucinated memory ID downgrades to recompute rather than trusting a fabricated
reference. I evaluated memory separately from the accuracy ablation on purpose —
mixing cache hits into an accuracy comparison would confound it."

### Q. How is any of this reproducible? LLMs are non-deterministic and CI can't call paid APIs.
**What they're testing:** engineering maturity around a genuinely hard problem.
**Answer:** "I separate three kinds of reproducibility. Prompt-exact cassettes are
keyed on a hash of role + system prompt + user prompt, so unchanged calls replay
exactly and prompt edits create intentional misses. Ordered scripted responses make
the orchestration, API, and WebSocket tests deterministic without keys or Docker.
Finally, committed raw benchmark outputs can be objectively re-scored at $0; an
integrity gate accepts only complete, zero-error runs and records their provenance.
That distinction matters: mocked CI proves the software path, while the saved live
run is the evidence for model quality."

### Q. What would you change? What are the limitations?
**What they're testing:** self-awareness. Never answer "nothing."
**Answer:** "Three honest ones. The deterministic layer now computes confidence
intervals, effect sizes, hypothesis tests, and direct standardization for categorical
confounders, but it does not yet fit causal/confounder-adjustment regressions for
continuous or high-cardinality confounding. Its predictive regressions estimate
out-of-sample accuracy, which is a different estimand. The published live scorecard predates that stratified
recomputation, so I keep the old trap score visible until a fresh live run measures
the gain. And the trap tier contains only five questions, so its point estimate is
directional until I repeat the selected policy across seeds."

### Q. How would you take this to production?
**What they're testing:** systems thinking beyond localhost.
**Answer:** "It's already structured for it — Docker Compose orchestrates the API, an
arq worker, Redis for the queue and pub/sub, Postgres for run history and traces, and
Jaeger for OpenTelemetry tracing across the agent graph. To go to Kubernetes the main
changes are: the worker becomes a horizontally-scalable Deployment, Redis/Postgres
become managed services or StatefulSets, and the Docker-in-Docker sandbox becomes the
real security question — that's where you'd graduate from per-execution containers to
gVisor or Firecracker microVMs for hard multi-tenant isolation. I called out the
docker-socket mount as root-equivalent in the compose file so it's not a hidden risk."

### Q. What does a run cost, and how do you control it?
**What they're testing:** cost-consciousness.
**Answer:** "I instrument tokens per agent step through the tracing layer, so cost is
observable per run and per agent, not a mystery. Control comes from three levers:
model routing (cheap model for mechanical steps), per-loop budgets that cap retries
and revisions, and the memory layer skipping recomputation for repeat questions. The
scorecard reports mean tokens per config—that is how I found that Standard costs
3,429 mean tokens, High assurance costs 17,223, and the adaptive router costs 19,892
without improving the Standard score. I turned those measurements into explicit
product modes instead of hiding the tradeoff."

### Q. What was the hardest part?
**What they're testing:** real engineering, told as a story.
**Answer options (pick one you can tell vividly):**
- *The concurrency model:* "Getting parallel executors to write shared state without
  locks — the insight was constraining them to disjoint keys with per-field reducers."
- *The free-tier eval:* "Running a ~600-request evaluation on free-tier APIs. I hit
  daily quota walls, so I built fail-fast quota detection and a resumable harness that
  skips already-passed questions, then recorded cassettes so re-runs cost nothing."
- *The 8B hallucination:* "An 8B executor confidently invented column names on
  multi-table joins — which is exactly why the eval measures the model-routing floor
  instead of me guessing it."

---

## Framing tips

- **Lead with the eval, not the agent count.** The evaluation rigor is what separates
  this from a thousand LangChain demos.
- **Volunteer the weak spots.** The critic's zero overall gain and the council's
  roughly 5× Standard-path token cost make you look
  *more* credible, not less — you measured your own system and didn't flinch.
- **Have one number per component memorized:** High assurance 0.86 vs Standard 0.80
  vs bare 0.56; 17,223 vs 3,429 mean tokens; critic recall 0.83 / false-positive 0.50.
- **Know what you deliberately didn't build** (Firecracker, interval recomputation)
  and why — scoping is a senior signal.
