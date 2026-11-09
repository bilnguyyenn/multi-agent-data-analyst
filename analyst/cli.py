"""CLI entry point for phase (a).

  analyst gen-data --out data/sample --seed 7
  analyst ask "Which machine had the most downtime hours?" --data data/sample \
      --sandbox docker --provider local
"""
from __future__ import annotations

import argparse
import json
import os
import uuid


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="analyst")
    sub = p.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="answer a business question against a dataset")
    ask.add_argument("question")
    ask.add_argument("--data", required=True, help="directory of CSV files")
    ask.add_argument("--sandbox", default=None, choices=["docker", "subprocess"])
    ask.add_argument(
        "--provider", default=None, choices=["local", "hosted", "replay", "anthropic"]
    )
    ask.add_argument("--timeout", type=int, default=60, help="sandbox seconds per task")
    critic = ask.add_mutually_exclusive_group()
    critic.add_argument(
        "--critic", dest="critic_enabled", action="store_true",
        help="enable the legacy adversarial critic (off by default after ablation)",
    )
    critic.add_argument(
        "--no-critic", dest="critic_enabled", action="store_false",
        help=argparse.SUPPRESS,
    )
    ask.set_defaults(critic_enabled=False)
    ask.add_argument(
        "--no-validator", action="store_true",
        help="disable the statistical validator (the eval ablation mode)",
    )
    ask.add_argument(
        "--no-council", action="store_true",
        help="disable the parallel analysis council and evidence adjudicator",
    )
    ask.add_argument(
        "--council-mode", choices=["adaptive", "always", "off"], default="off",
        help="review policy: off=standard (default), always=high assurance, "
             "adaptive=experimental",
    )
    ask.add_argument(
        "--no-memory", action="store_true",
        help="disable the vector memory layer (the eval ablation mode)",
    )
    ask.add_argument("--json", action="store_true", help="emit full final state as JSON")

    gen = sub.add_parser("gen-data", help="generate the synthetic manufacturing dataset")
    gen.add_argument("--out", default="data/sample")
    gen.add_argument("--seed", type=int, default=7)

    heldout_gen = sub.add_parser(
        "gen-heldout-data",
        help="generate the independent product-analytics statistical benchmark data",
    )
    heldout_gen.add_argument("--out", default=".heldout_data/seed_20260717")
    heldout_gen.add_argument("--seed", type=int, default=20260717)

    ev = sub.add_parser("eval", help="run the benchmark + ablations, emit scorecard")
    ev.add_argument("--data", default="data/sample")
    ev.add_argument("--provider", default=None,
                    choices=["local", "hosted", "replay", "anthropic"])
    ev.add_argument("--configs", default="council-no-critic,no-critic",
                    help="comma list from: adaptive-council,full-council,"
                         "council-no-critic,full,no-critic,no-validator,bare")
    ev.add_argument("--subset", default="all", choices=["all", "smoke"])
    ev.add_argument("--critic-bench", action="store_true",
                    help="also run the isolated flawed-specimen critic benchmark")
    ev.add_argument("--resume", action="store_true",
                    help="skip questions that already passed in eval_results/, and "
                         "re-attempt only the errored/missing ones (use after a "
                         "quota-killed or interrupted run)")
    ev.add_argument("--timeout", type=int, default=60)

    sub.add_parser(
        "scorecard",
        help="rebuild the scorecard from the newest complete, zero-error saved runs",
    )

    heldout = sub.add_parser(
        "heldout-eval",
        help="run the separate 20-question held-out statistical benchmark",
    )
    heldout.add_argument("--data-root", default=".heldout_data")
    heldout.add_argument("--out-dir", default="heldout_eval_results")
    heldout.add_argument("--seeds", default="20260717", help="comma-separated integer seeds")
    heldout.add_argument("--configs", default="no-critic", help="comma-separated eval configs")
    heldout.add_argument("--provider", default=None,
                         choices=["local", "hosted", "replay", "anthropic"])
    heldout.add_argument("--subset", default="all", choices=["all", "smoke"])
    heldout.add_argument("--resume", action="store_true")
    heldout.add_argument("--timeout", type=int, default=60)

    heldout_scorecard = sub.add_parser(
        "heldout-scorecard",
        help="rebuild the held-out scorecard from complete zero-error saved runs",
    )
    heldout_scorecard.add_argument("--out-dir", default="heldout_eval_results")

    predictive = sub.add_parser(
        "predictive-benchmark", help="run the API-free predictive reliability suite (fixed seeds)",
    )
    predictive.add_argument("--out-dir", default="predictive_eval_results")

    export = sub.add_parser("export-report", help="export a completed run JSON as a portable report")
    export.add_argument("input", help="saved run JSON or the output of ask --json")
    export.add_argument("--out", required=True, help="destination .html or .json file")
    export.add_argument("--format", choices=["html", "json"], default=None)

    tv = sub.add_parser("trap-variance",
                        help="repeat only the tier-4 traps K times for a 95%% CI on "
                             "the noisy trap column")
    tv.add_argument("--data", default="data/sample")
    tv.add_argument("--provider", default=None,
                    choices=["local", "hosted", "replay", "anthropic"])
    tv.add_argument("--config", default="council-no-critic",
                    help="one benchmark config (default: selected high-assurance policy)")
    tv.add_argument("--repeats", type=int, default=5)
    tv.add_argument("--timeout", type=int, default=60)

    args = p.parse_args(argv)

    if args.cmd == "export-report":
        from pathlib import Path

        from analyst.reporting import render_report, report_payload

        try:
            run = json.loads(Path(args.input).read_text())
            payload = report_payload(run)
            format_name = args.format or ("json" if Path(args.out).suffix.lower() == ".json" else "html")
            content = (render_report(payload) if format_name == "html"
                       else json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            Path(args.out).write_text(content)
        except (OSError, ValueError, TypeError) as exc:
            print(f"could not export report: {exc}")
            return 1
        print(f"report: {args.out}")
        return 0

    if args.cmd == "predictive-benchmark":
        from analyst.eval.predictive_benchmark import run_benchmark, write_report

        raw, report, passed = write_report(run_benchmark(), args.out_dir)
        print(f"predictive reliability: {'PASS' if passed else 'FAIL'}")
        print(f"raw evidence: {raw}")
        print(f"report: {report}")
        return 0 if passed else 1

    if args.cmd == "gen-data":
        from analyst.eval.data_gen import generate

        paths = generate(args.out, seed=args.seed)
        print(f"Wrote {len(paths)} tables to {args.out}")
        return 0

    if args.cmd == "gen-heldout-data":
        from analyst.eval.heldout_data import generate_heldout

        paths = generate_heldout(args.out, seed=args.seed)
        print(f"Wrote {len(paths) - 1} held-out tables to {args.out}")
        return 0

    if args.cmd == "eval":
        if args.provider:
            os.environ["ANALYST_LLM_PROVIDER"] = args.provider
        # Record every live call as a cassette: a re-run of anything already
        # answered is free. (A 6-hour run was once lost to an end-of-run crash;
        # every safeguard in this block exists because of it.)
        os.environ.setdefault("ANALYST_RECORD_CASSETTES", "1")
        from analyst.eval.harness import (
            load_prior,
            run_config,
            save_raw,
            select_latest_valid_runs,
        )
        from analyst.eval.scorecard import write as write_scorecard

        runs = []
        quota_hit = False
        for name in args.configs.split(","):
            name = name.strip()
            prior = load_prior(name) if args.resume else {}
            if prior:
                print(f"   resume: {len(prior)} question(s) already passed for "
                      f"{name} — will skip them", flush=True)
            print(f"== config: {name} ({args.subset}) ==", flush=True)
            try:
                run = run_config(name, args.data, subset=args.subset,
                                 provider=args.provider, timeout=args.timeout,
                                 prior=prior)
            except Exception as e:  # one config dying must not lose the others
                print(f"!! config {name} aborted: {e}", flush=True)
                continue
            runs.append(run)
            p = save_raw([run])  # SAVE IMMEDIATELY, one file per config
            print(f"   saved: {p}", flush=True)
            if run.get("aborted"):  # daily quota gone — don't start more configs
                quota_hit = True
                print("== stopping matrix: daily quota exhausted. Re-run the same "
                      "command with --resume after midnight Pacific. ==", flush=True)
                break

        bench = None
        if args.critic_bench and not quota_hit:
            from analyst.data import build_manifest
            from analyst.eval.critic_bench import run_critic_bench
            from analyst.llm.client import make_client

            print("== critic-bench ==", flush=True)
            try:
                bench = run_critic_bench(make_client(provider=args.provider),
                                         build_manifest(args.data))
                from pathlib import Path

                bench_path = Path("eval_results/critic_bench.json")
                bench_path.parent.mkdir(exist_ok=True)
                bench_path.write_text(json.dumps(bench, indent=2))
            except Exception as e:
                print(f"!! critic-bench aborted: {e} (scorecard still written)",
                      flush=True)

        if runs or bench:
            selected, sources, rejected = select_latest_valid_runs()
            if selected:
                if bench is None:
                    from pathlib import Path

                    bench_path = Path("eval_results/critic_bench.json")
                    if bench_path.exists():
                        bench = json.loads(bench_path.read_text())
                card = write_scorecard(
                    list(selected.values()), bench, sources=sources
                )
                print(f"scorecard: {card}")
                if rejected:
                    print(
                        f"integrity gate ignored {len(rejected)} broken/partial "
                        "saved run(s)"
                    )
            else:
                print("scorecard unchanged: no complete, zero-error benchmark runs")
        return 0

    if args.cmd == "trap-variance":
        if args.provider:
            os.environ["ANALYST_LLM_PROVIDER"] = args.provider
        # We need FRESH samples each repeat, so don't record (protects the golden
        # cassettes from the main eval) and don't replay (that would be identical
        # every run — zero variance). Force live calls.
        os.environ.pop("ANALYST_RECORD_CASSETTES", None)
        from analyst.eval.harness import run_trap_variance, write_trap_variance

        print(f"== trap-variance: {args.config}, {args.repeats} repeats ==",
              flush=True)
        summ = run_trap_variance(args.config, args.data, provider=args.provider,
                                 repeats=args.repeats, timeout=args.timeout)
        path = write_trap_variance(summ)
        c = summ["tier4_correct"]
        print(f"\ntier-4 correctness: {c['mean']:.3f} ± {c['ci95_halfwidth']:.3f} "
              f"(95% CI over {summ['repeats']} runs)")
        print(f"report: {path}")
        return 0

    if args.cmd == "heldout-eval":
        from pathlib import Path

        from analyst.data import build_manifest
        from analyst.eval.heldout_data import generate_heldout
        from analyst.eval.heldout_harness import (
            load_heldout_prior,
            run_heldout_config,
            save_heldout_raw,
            select_latest_valid_heldout_runs,
            write_heldout_scorecard,
        )

        if args.provider:
            os.environ["ANALYST_LLM_PROVIDER"] = args.provider
        os.environ.setdefault("ANALYST_RECORD_CASSETTES", "1")
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        configs = [value.strip() for value in args.configs.split(",") if value.strip()]
        quota_hit = False
        for seed in seeds:
            data_dir = Path(args.data_root) / f"seed_{seed}"
            generate_heldout(data_dir, seed=seed)
            dataset_hash = build_manifest(data_dir).content_hash
            for config_name in configs:
                prior = (
                    load_heldout_prior(
                        config_name, seed, dataset_hash, out_dir=args.out_dir
                    )
                    if args.resume
                    else {}
                )
                print(
                    f"== held-out config: {config_name} · seed {seed} ({args.subset}) ==",
                    flush=True,
                )
                run = run_heldout_config(
                    config_name,
                    data_dir,
                    seed,
                    subset=args.subset,
                    provider=args.provider,
                    timeout=args.timeout,
                    prior=prior,
                )
                raw_path = save_heldout_raw([run], out_dir=args.out_dir)
                print(f"   saved: {raw_path}", flush=True)
                if run.get("aborted"):
                    quota_hit = True
                    break
            if quota_hit:
                break
        selected, sources, rejected = select_latest_valid_heldout_runs(args.out_dir)
        if selected:
            scorecard = write_heldout_scorecard(
                list(selected.values()), sources=sources, out_dir=args.out_dir
            )
            print(f"held-out scorecard: {scorecard}")
            if rejected:
                print(f"integrity gate ignored {len(rejected)} broken/partial run(s)")
        else:
            print("held-out scorecard unchanged: no complete, zero-error runs")
        return 0

    if args.cmd == "heldout-scorecard":
        from analyst.eval.heldout_harness import (
            select_latest_valid_heldout_runs,
            write_heldout_scorecard,
        )

        selected, sources, rejected = select_latest_valid_heldout_runs(args.out_dir)
        if not selected:
            print(f"no complete, zero-error held-out runs in {args.out_dir}/")
            return 1
        path = write_heldout_scorecard(
            list(selected.values()), sources=sources, out_dir=args.out_dir
        )
        print(f"merged runs: {len(selected)}\nheld-out scorecard: {path}")
        if rejected:
            print(f"integrity gate ignored {len(rejected)} broken/partial run(s)")
        return 0

    if args.cmd == "scorecard":
        # Rebuild from the latest VALID run per config. A failed replay or partial
        # smoke run is retained for diagnosis but cannot poison published metrics.
        from pathlib import Path

        from analyst.eval.harness import select_latest_valid_runs
        from analyst.eval.scorecard import write as write_scorecard

        latest, sources, rejected = select_latest_valid_runs()
        if not latest:
            print("no complete, zero-error benchmark runs in eval_results/")
            return 1
        bench_path = Path("eval_results/critic_bench.json")
        bench = json.loads(bench_path.read_text()) if bench_path.exists() else None
        card = write_scorecard(list(latest.values()), bench, sources=sources)
        print(f"merged configs: {sorted(latest)}\nscorecard: {card}")
        if rejected:
            print(f"integrity gate ignored {len(rejected)} broken/partial saved run(s)")
        return 0

    # ---- ask ----
    if args.sandbox:
        os.environ["ANALYST_SANDBOX"] = args.sandbox
    if args.provider:
        os.environ["ANALYST_LLM_PROVIDER"] = args.provider

    from analyst.data import build_manifest
    from analyst.graph.control_loop import build_graph
    from analyst.graph.state import Budgets
    from analyst.llm.client import make_client
    from analyst.sandbox.runner import make_runner

    llm = make_client()
    sandbox = make_runner()
    manifest = build_manifest(args.data)
    memory = None
    if not args.no_memory:
        from analyst.memory.store import MemoryStore

        memory = MemoryStore()

    graph = build_graph(
        llm, sandbox, exec_timeout_s=args.timeout,
        critic_enabled=args.critic_enabled,
        validator_enabled=not args.no_validator,
        council_mode="off" if args.no_council else args.council_mode,
        memory=memory,
    )

    state = graph.invoke(
        {
            "run_id": uuid.uuid4().hex[:12],
            "question": args.question,
            "dataset": manifest,
            "budgets": Budgets(),
        },
        config={"recursion_limit": 120},
    )

    if args.json:
        out = dict(state)
        out["status"] = "done"
        out["dataset"] = state["dataset"].model_dump()
        out["dag"] = {k: v.model_dump() for k, v in state.get("dag", {}).items()}
        out["artifacts"] = {
            k: v.model_dump() for k, v in state.get("artifacts", {}).items()
        }
        out["budgets"] = state["budgets"].model_dump() if state.get("budgets") else None
        out["council_findings"] = {
            k: v.model_dump() for k, v in state.get("council_findings", {}).items()
        }
        out["council_decisions"] = [
            decision.model_dump() for decision in state.get("council_decisions", [])
        ]
        out["council_routing"] = (
            state["council_routing"].model_dump()
            if state.get("council_routing") else None
        )
        out["statistical_evidence"] = {
            k: value.model_dump()
            for k, value in state.get("statistical_evidence", {}).items()
        }
        out["predictive_evidence"] = {
            k: value.model_dump()
            for k, value in state.get("predictive_evidence", {}).items()
        }
        print(json.dumps(out, indent=2, default=str))
        return 0

    # Always persist the full run state for post-mortems — reruns are expensive
    # on an 8GB machine, debugging from a file is free.
    from pathlib import Path

    from analyst.reporting import dataset_provenance

    dump_dir = Path(".analyst_runs")
    dump_dir.mkdir(exist_ok=True)
    (dump_dir / "last_run.json").write_text(
        json.dumps(
            {
                "run_id": state.get("run_id"),
                "status": "done",
                "question": state.get("question"),
                "answer": state.get("answer"),
                "dag": {k: v.model_dump() for k, v in state.get("dag", {}).items()},
                "artifacts": {
                    k: v.model_dump() for k, v in state.get("artifacts", {}).items()
                },
                "events": state.get("events", []),
                "council_findings": {
                    k: v.model_dump()
                    for k, v in state.get("council_findings", {}).items()
                },
                "council_decisions": [
                    decision.model_dump()
                    for decision in state.get("council_decisions", [])
                ],
                "council_routing": (
                    state["council_routing"].model_dump()
                    if state.get("council_routing") else None
                ),
                "data_quality_report": (
                    state["dataset"].quality_report.model_dump()
                    if state.get("dataset") and state["dataset"].quality_report
                    else None
                ),
                "dataset_provenance": dataset_provenance(state.get("dataset")),
                "claims": [claim.model_dump() for claim in state.get("claims", [])],
                "statistical_evidence": {
                    key: value.model_dump()
                    for key, value in state.get("statistical_evidence", {}).items()
                },
                "predictive_evidence": {
                    key: value.model_dump()
                    for key, value in state.get("predictive_evidence", {}).items()
                },
            },
            indent=2,
            default=str,
        )
    )

    print("=" * 72)
    print("ANSWER")
    print("=" * 72)
    print(state.get("answer", "(none)"))
    print()
    quality_report = state["dataset"].quality_report if state.get("dataset") else None
    if quality_report is not None:
        counts = quality_report.issue_counts
        print(
            f"data quality: {quality_report.quality_score:.1f}/100 · "
            f"{counts.get('error', 0)} errors · {counts.get('warning', 0)} warnings · "
            f"{counts.get('info', 0)} EDA signals"
        )
        for issue in quality_report.issues[:5]:
            location = f"{issue.table}.{issue.column}" if issue.column else issue.table
            print(f"  [{issue.severity}] {location}: {issue.message}")
        print()
    print("tasks:")
    for t in state.get("dag", {}).values():
        art = state.get("artifacts", {}).get(t.id)
        toks = f"{art.input_tokens + art.output_tokens} tok" if art else "-"
        dur = f"{art.outcome.duration_s:.1f}s" if art else "-"
        # total executions = crash retries + critic-ordered rewrites + final success
        runs = t.attempts + t.revisions + (1 if t.status == "done" else 0)
        rev = f", revised x{t.revisions}" if t.revisions else ""
        print(f"  [{t.status:>7}] {t.id} ({t.kind}, runs={runs}{rev}, {dur}, {toks}) {t.goal}")
        if t.status == "failed" and art:
            tail = (art.outcome.stderr or art.outcome.stdout or "(no output)").strip()
            print(f"           last failure ({art.outcome.status}): {tail[-300:]}")
    print("\nfull state saved to .analyst_runs/last_run.json")
    total = sum(
        a.input_tokens + a.output_tokens for a in state.get("artifacts", {}).values()
    )
    print(f"\ntotal executor tokens: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
