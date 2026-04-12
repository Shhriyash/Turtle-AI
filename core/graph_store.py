from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class GraphContextQuery:
    query: str
    task_type: str
    max_lines: int = 4


class GraphStore:
    """DEPRECATED: legacy derived graph memory.

    This store remains only for compatibility with the JSON fallback memory path.
    Primary personalization now lives in Markdown personal memory files.
    """

    def __init__(self, graph_path: Path):
        self.graph_path = graph_path

    def load_graph(self) -> dict[str, Any]:
        if not self.graph_path.exists():
            return self._default_graph()
        try:
            payload = json.loads(self.graph_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return self._default_graph()

    def save_graph(self, graph: dict[str, Any]) -> None:
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph["meta"] = graph.get("meta", {})
        graph["meta"]["updated_at"] = _utc_now()
        self.graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")

    def rebuild_from_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        graph = self._default_graph()
        self._add_node(graph, "user:self", "user", profile.get("identity", {}).get("name") or "User")

        identity = profile.get("identity", {})
        preferences = profile.get("preferences", {})
        workflow = profile.get("workflow", {})
        tools = profile.get("tool_preferences", {})

        for email in identity.get("emails", []):
            email_id = f"email:{email.lower()}"
            self._add_node(graph, email_id, "email", email)
            self._add_edge(graph, "user:self", email_id, "has_email", 1.0, 1.0)

        if identity.get("timezone"):
            tz = str(identity["timezone"]).strip()
            tz_id = f"timezone:{tz}"
            self._add_node(graph, tz_id, "timezone", tz)
            self._add_edge(graph, "user:self", tz_id, "in_timezone", 1.0, 1.0)

        for key in ["response_style", "humor_level", "email_tone"]:
            value = preferences.get(key)
            if not value:
                continue
            pref_id = f"pref:{key}:{str(value).lower()}"
            self._add_node(graph, pref_id, "preference", f"{key}={value}")
            self._add_edge(graph, "user:self", pref_id, "prefers", 1.0, 1.0)

        for recipient in workflow.get("common_recipients", []):
            recipient_id = f"recipient:{str(recipient).lower()}"
            self._add_node(graph, recipient_id, "recipient", recipient)
            self._add_edge(graph, "user:self", recipient_id, "emails", 1.0, 0.7)

        for key in ["prefers_draft_before_send"]:
            value = workflow.get(key)
            if value is None:
                continue
            wf_id = f"workflow:{key}:{str(value).lower()}"
            self._add_node(graph, wf_id, "workflow", f"{key}={value}")
            self._add_edge(graph, "user:self", wf_id, "workflow_pref", 1.0, 0.75)

        if tools.get("primary_llm"):
            model = str(tools["primary_llm"]).strip()
            model_id = f"tool_model:{model.lower()}"
            self._add_node(graph, model_id, "tool", model)
            self._add_edge(graph, "user:self", model_id, "uses", 1.0, 0.9)

        return graph

    def query_context(self, query: GraphContextQuery) -> list[str]:
        graph = self.load_graph()
        edges = graph.get("edges", [])
        nodes = {node.get("id"): node for node in graph.get("nodes", []) if isinstance(node, dict)}
        query_text = (query.query or "").lower()
        lines: list[str] = []

        for edge in edges:
            edge_type = str(edge.get("type", "")).lower()
            target_id = edge.get("target")
            target = nodes.get(target_id, {})
            label = str(target.get("label", "")).strip()
            if not label:
                continue

            if query.task_type == "email" and edge_type in {"emails", "has_email", "workflow_pref", "prefers"}:
                lines.append(f"Relationship: {edge_type} {label}")
            elif query.task_type == "general" and edge_type in {"prefers", "workflow_pref"}:
                lines.append(f"Relationship: {edge_type} {label}")
            elif any(token in query_text for token in ["usually", "prefer", "often", "normal", "habit"]):
                lines.append(f"Relationship: {edge_type} {label}")

            if len(lines) >= query.max_lines:
                break

        return lines

    @staticmethod
    def _default_graph() -> dict[str, Any]:
        return {
            "nodes": [],
            "edges": [],
            "meta": {"version": 1, "updated_at": _utc_now()},
        }

    @staticmethod
    def _add_node(graph: dict[str, Any], node_id: str, node_type: str, label: str) -> None:
        nodes = graph.setdefault("nodes", [])
        for node in nodes:
            if node.get("id") == node_id:
                return
        nodes.append({"id": node_id, "type": node_type, "label": label})

    @staticmethod
    def _add_edge(
        graph: dict[str, Any],
        source: str,
        target: str,
        edge_type: str,
        weight: float,
        confidence: float,
    ) -> None:
        edges = graph.setdefault("edges", [])
        for edge in edges:
            if edge.get("source") == source and edge.get("target") == target and edge.get("type") == edge_type:
                edge["weight"] = max(float(edge.get("weight", 0.0)), weight)
                edge["confidence"] = max(float(edge.get("confidence", 0.0)), confidence)
                edge["updated_at"] = _utc_now()
                return
        edges.append(
            {
                "source": source,
                "target": target,
                "type": edge_type,
                "weight": weight,
                "confidence": confidence,
                "updated_at": _utc_now(),
            }
        )
