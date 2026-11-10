# Counterbalanced retail usability pilot — prepared, not conducted

Actual participants were deferred by the user. `SUMMARY.json` must therefore
remain pending until real observations exist. Never substitute simulated trials.

1. Recruit 3–5 willing volunteers yourself. Use aliases P01, P02, etc. Explain the
   purpose, voluntary nature, and ability to withdraw. Obtain consent before
   recording timings. No names, email addresses, recordings, or quotations are needed.
2. Assign AB, BA, AB, BA in signup order. There are three matched task pairs,
   one on each public weekly CSV. AB uses manual tools for variant A and Analyst
   for B; BA reverses conditions. Alternate which condition is performed first.
3. Give participants `participant-tasks.json`, the data, metric definitions in
   `docs/OPERATIONS.md`, and the same short setup practice. Keep facilitator gold
   answers separate. Manual means spreadsheet/SQL/pandas without Analyst or an LLM.
4. Time each task from presenting the question until submission. Record the answer,
   elapsed seconds, abandoned attempts, and interventions. Incorrect/abandoned
   tasks stay in the trial set. Do not cherry-pick successful times.
5. Ask whether the participant would reuse the workflow and what was confusing.
   Record a boolean plus optional non-identifying notes; obtain separate permission
   before publishing a quote. A small volunteer study is not enterprise adoption.

Prepare: `python -m analyst retail-pilot`.
Analyze: `python -m analyst retail-pilot --analyze`.

The ignored local `observations.json` starts with empty participants/trials.
Populate it with this schema (the example describes fields, not observations):

```text
protocol_hash: exact value from facilitator-protocol.json
participants: [{id: alias, consent: true, sequence_group: AB or BA, would_reuse: boolean}]
trials: [{participant: alias, task_id: pair1A etc., condition: manual or analyst,
          elapsed_seconds: measured positive seconds, answer: number or null,
          interventions: nonnegative integer, synthetic: false}]
```

Every consented participant needs all six trial entries for a complete pilot.
An unanswered task has `answer: null`, retains its elapsed time and interventions,
and counts as incorrect. The analyzer rejects synthetic trials, duplicates,
unknown IDs, invalid durations, protocol changes, and counterbalance violations.
It reports all trials, correctness, interventions, and median time among correct
answers separately. With fewer than three participants or missing trials, status
is incomplete. No broad causal claim about time savings is supported by this design.

Results should be reviewed before publishing. Raw local observations are ignored
by Git; publish only consented, non-identifying aggregates and the protocol.
