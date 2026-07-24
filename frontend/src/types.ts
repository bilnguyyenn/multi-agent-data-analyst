// Mirrors the Python event protocol (analyst/api/runner.py) — one source of
// truth per side, kept in sync by the API e2e tests.

export type TaskStatus =
  | "pending" | "running" | "review" | "done" | "rejected" | "failed";

export interface TaskInfo {
  id: string;
  goal: string;
  kind: "python" | "sql" | "reuse";
  status: TaskStatus;
  depends_on: string[];
  attempts: number;
  revisions: number;
  reuse_of: string | null;
}

export interface AgentEvent {
  ts: number;
  node: string;
  [key: string]: unknown;
}

export interface Claim {
  task_id: string;
  claim: string;
  verdict: "pass" | "caveat" | "fail";
  note: string;
}

export interface ConfidenceInterval {
  parameter: string;
  estimate: number;
  lower: number;
  upper: number;
  level: number;
  method: string;
}

export interface EffectSize {
  name: string;
  value: number;
  interpretation: string;
}

export interface HypothesisTest {
  name: string;
  statistic: number;
  p_value: number;
  alternative: "two-sided" | "less" | "greater";
  null_value: number | null;
  alpha: number;
  significant: boolean;
}

export interface StatisticalEvidence {
  task_id: string;
  request_id: string;
  kind: string;
  status: "computed" | "invalid";
  sample_size: number;
  intervals: ConfidenceInterval[];
  effects: EffectSize[];
  test: HypothesisTest | null;
  summary: string;
  warnings: string[];
  diagnostics: Record<string, unknown>;
}

export interface PredictiveMetric {
  name: string;
  value: number;
  higher_is_better: boolean;
  baseline_value: number | null;
  lower: number | null;
  upper: number | null;
  confidence: number | null;
}

export interface ModelCandidate {
  name: string;
  cv_mean: number;
  cv_std: number;
  folds: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  std: number;
}

export interface SliceMetric {
  column: string;
  value: string;
  n: number;
  metric: string;
  score: number | null;
  warning: string;
}

export interface PredictiveEvidence {
  task_id: string;
  request_id: string;
  task_type: string;
  status: "computed" | "invalid";
  target: string;
  features: string[];
  selected_model: string;
  primary_metric: string;
  train_size: number;
  test_size: number;
  split_strategy: string;
  metrics: PredictiveMetric[];
  candidates: ModelCandidate[];
  feature_importance: FeatureImportance[];
  slices: SliceMetric[];
  summary: string;
  warnings: string[];
  diagnostics: Record<string, unknown>;
}

export interface ValueFrequency {
  value: string;
  count: number;
  rate: number;
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  semantic_type: "numeric" | "categorical" | "text" | "datetime" | "boolean" | "identifier" | "unknown";
  row_count: number;
  non_null_count: number;
  missing_count: number;
  missing_rate: number;
  distinct_count: number;
  distinct_rate: number;
  numeric_summary: Record<string, number>;
  datetime_summary: Record<string, string>;
  top_values: ValueFrequency[];
}

export interface TableProfile {
  name: string;
  row_count: number;
  column_count: number;
  duplicate_rows: number;
  duplicate_rate: number;
  candidate_keys: string[];
  columns: ColumnProfile[];
}

export interface QualityIssue {
  code: string;
  severity: "error" | "warning" | "info";
  table: string;
  column: string | null;
  message: string;
  recommendation: string;
  affected_count: number;
  affected_rate: number;
}

export interface RelationshipProfile {
  column: string;
  parent_table: string;
  child_table: string;
  child_non_null_count: number;
  orphan_count: number;
  orphan_rate: number;
}

export interface DataQualityReport {
  quality_score: number;
  score_method: string;
  table_count: number;
  total_rows: number;
  issue_counts: Record<"error" | "warning" | "info", number>;
  summary: string;
  tables: TableProfile[];
  relationships: RelationshipProfile[];
  issues: QualityIssue[];
}

export interface CouncilFinding {
  specialist: "metric" | "statistical" | "data_quality";
  round: number;
  verdict: "support" | "challenge" | "insufficient";
  conclusion: string;
  answer_key: string;
  evidence_refs: string[];
  risks: string[];
  confidence: number;
  recommended_action: "accept" | "recompute";
  target_task_id: string | null;
  instruction: string;
}

export interface CouncilDecision {
  round: number;
  status: "consensus" | "adjudicated" | "recompute" | "caveat";
  conclusion: string;
  rationale: string;
  dissent: string[];
  action: "accept" | "recompute";
  task_id: string | null;
  instruction: string;
}

export interface CouncilRoutingDecision {
  round: number;
  mode: "off" | "adaptive" | "always";
  invoked: boolean;
  risk_score: number;
  threshold: number;
  reasons: string[];
}

export interface RunSummary {
  run_id: string;
  question: string;
  status: string;
  created_at: number;
  answer: string | null;
}

export type WsMessage =
  | { type: "replay"; run: { status: string; answer: string | null; dag: Record<string, TaskInfo>; events: AgentEvent[]; claims: Claim[]; statistical_evidence?: StatisticalEvidence[]; predictive_evidence?: PredictiveEvidence[]; data_quality_report?: DataQualityReport | null; council_findings?: CouncilFinding[]; council_decisions?: CouncilDecision[]; council_routing?: CouncilRoutingDecision | null } }
  | { type: "status"; status: string; detail?: string }
  | { type: "node_update"; tasks: TaskInfo[]; new_events: AgentEvent[]; statistical_evidence?: StatisticalEvidence[]; predictive_evidence?: PredictiveEvidence[]; data_quality_report?: DataQualityReport | null; council_findings?: CouncilFinding[]; council_decisions?: CouncilDecision[]; council_routing?: CouncilRoutingDecision | null }
  | { type: "answer"; answer: string; claims: Claim[]; statistical_evidence?: StatisticalEvidence[]; predictive_evidence?: PredictiveEvidence[]; data_quality_report?: DataQualityReport | null; council_findings?: CouncilFinding[]; council_decisions?: CouncilDecision[]; council_routing?: CouncilRoutingDecision | null }
  | { type: "stream_end" };
