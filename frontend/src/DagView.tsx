// Live DAG: the planner's task graph, statuses updating as agents work.
// Layout is computed from depends_on with plain topological layering — no
// layout library needed at 1-5 nodes.
import { ReactFlow, Background, type Node, type Edge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { TaskInfo } from "./types";

const STATUS_COLOR: Record<string, string> = {
  pending: "#9ca3af",
  running: "#3b82f6",
  review: "#a855f7",   // purple: awaiting the critic
  done: "#22c55e",
  rejected: "#f97316", // orange: critic forced a replan
  failed: "#ef4444",
};

function layers(tasks: TaskInfo[]): Map<string, number> {
  const level = new Map<string, number>();
  const byId = new Map(tasks.map((t) => [t.id, t]));
  const depth = (id: string, seen: Set<string>): number => {
    if (level.has(id)) return level.get(id)!;
    if (seen.has(id)) return 0; // cycle guard (shouldn't happen; planner validates)
    seen.add(id);
    const t = byId.get(id);
    const d = t && t.depends_on.length
      ? 1 + Math.max(...t.depends_on.map((p) => depth(p, seen)))
      : 0;
    level.set(id, d);
    return d;
  };
  tasks.forEach((t) => depth(t.id, new Set()));
  return level;
}

export default function DagView({ tasks }: { tasks: TaskInfo[] }) {
  const level = layers(tasks);
  const perLevelIndex = new Map<number, number>();

  const nodes: Node[] = tasks.map((t) => {
    const l = level.get(t.id) ?? 0;
    const idx = perLevelIndex.get(l) ?? 0;
    perLevelIndex.set(l, idx + 1);
    const badge =
      t.kind === "reuse" ? " ♻" : t.revisions > 0 ? ` ✎${t.revisions}` : "";
    return {
      id: t.id,
      position: { x: 40 + l * 260, y: 40 + idx * 110 },
      data: { label: `${t.id}${badge} · ${t.status}\n${t.goal.slice(0, 60)}` },
      style: {
        border: `2px solid ${STATUS_COLOR[t.status] ?? "#9ca3af"}`,
        borderRadius: 8,
        padding: 8,
        width: 220,
        fontSize: 11,
        whiteSpace: "pre-wrap",
        background: "white",
        boxShadow: t.status === "running" ? `0 0 12px ${STATUS_COLOR.running}` : "none",
      },
    };
  });

  const edges: Edge[] = tasks.flatMap((t) =>
    t.depends_on.map((dep) => ({
      id: `${dep}->${t.id}`,
      source: dep,
      target: t.id,
      animated: t.status === "running",
    }))
  );

  return (
    <div style={{ height: "100%", minHeight: 320 }}>
      <ReactFlow nodes={nodes} edges={edges} fitView proOptions={{ hideAttribution: true }}>
        <Background />
      </ReactFlow>
    </div>
  );
}
