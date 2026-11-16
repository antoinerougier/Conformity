"""LangGraph compliance-audit graph.

```
START -> parse_document -> profile_project -> plan_queries -> retrieve -> rethink
                                                                 ^           |
                                                                 +-----------+ insufficient evidence
                                                                             | (<= MAX_RETHINK_ITERATIONS)
                                       +-------------------------------------+ sufficient evidence
                                       v                                     v
                               classify_ai_act                          audit_gdpr      (parallel)
                                       +------------------+------------------+
                                                          v
                                                     synthesize -> END
```
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, FilePath

from src.agents.nodes import NODE_LABELS, AuditDependencies, AuditNodes
from src.compliance.citations import CitationValidator
from src.core.config import Settings, get_settings
from src.core.exceptions import AuditExecutionError
from src.core.llm import LLMService, OllamaLLMService
from src.core.types import validated
from src.models.schemas import AuditReport, AuditState, ProjectDocument, ProjectProfile
from src.rag.retriever import HybridRetriever, build_retriever

AGENT_NODES = ["classify_ai_act", "audit_gdpr"]
_BASE_RECURSION_LIMIT = 12


def route_after_rethink(state: AuditState) -> str | list[str]:
    """Corrective loop if the Rethinker reformulated queries, otherwise fan out to the agents."""
    return "retrieve" if state.pending_queries else AGENT_NODES


def build_graph(deps: AuditDependencies) -> CompiledStateGraph[Any, Any, Any, Any]:
    nodes = AuditNodes(deps)
    graph: StateGraph[AuditState, Any, Any, Any] = StateGraph(AuditState)

    graph.add_node("parse_document", nodes.parse_document)
    graph.add_node("profile_project", nodes.profile_project)
    graph.add_node("plan_queries", nodes.plan_queries)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("rethink", nodes.rethink)
    graph.add_node("classify_ai_act", nodes.classify_ai_act)
    graph.add_node("audit_gdpr", nodes.audit_gdpr)
    graph.add_node("synthesize", nodes.synthesize)

    graph.add_edge(START, "parse_document")
    graph.add_edge("parse_document", "profile_project")
    graph.add_edge("profile_project", "plan_queries")
    graph.add_edge("plan_queries", "retrieve")
    graph.add_edge("retrieve", "rethink")
    graph.add_conditional_edges("rethink", route_after_rethink, ["retrieve", *AGENT_NODES])
    graph.add_edge(AGENT_NODES, "synthesize")  # join: waits for both agents
    graph.add_edge("synthesize", END)
    return graph.compile()


class ProgressEvent(BaseModel):
    """One executed node, as streamed to the UI."""

    model_config = ConfigDict(frozen=True)

    node: str
    label: str
    trace: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ComplianceAuditor:
    """High-level facade: synchronous or streamed execution (UI progress)."""

    def __init__(self, deps: AuditDependencies) -> None:
        self.deps = deps
        self.graph = build_graph(deps)

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        llm: LLMService | None = None,
        retriever: HybridRetriever | None = None,
    ) -> ComplianceAuditor:
        settings = settings or get_settings()
        retriever = retriever or build_retriever(settings)
        deps = AuditDependencies(
            llm=llm or OllamaLLMService(settings),
            retriever=retriever,
            validator=CitationValidator.from_corpus(retriever.bm25.chunks),
            settings=settings,
        )
        return cls(deps)

    def _config(self) -> RunnableConfig:
        # Each Rethinker iteration consumes 2 super-steps (retrieve + rethink).
        return {
            "recursion_limit": _BASE_RECURSION_LIMIT + 2 * (self.deps.settings.max_rethink_iterations + 1)
        }

    @staticmethod
    def _state_for_file(project_path: Path) -> AuditState:
        return AuditState(project_path=str(project_path))

    @staticmethod
    @validated
    def state_for_context(
        document: ProjectDocument, profile: ProjectProfile | None = None, interview_transcript: str = ""
    ) -> AuditState:
        """Initial state for an audit whose context was gathered by an assessment session."""
        return AuditState(
            project_path=document.file_name,
            file_name=document.file_name,
            document=document,
            profile=profile,
            interview_transcript=interview_transcript,
        )

    def invoke(self, initial: AuditState) -> AuditReport:
        result = self.graph.invoke(initial, config=self._config())
        report = result.get("report")
        if not isinstance(report, AuditReport):
            raise AuditExecutionError("The graph finished without producing a report")
        return report

    def iter_events(self, initial: AuditState) -> Iterator[ProgressEvent | AuditReport]:
        """Yield one :class:`ProgressEvent` per executed node, then the final report."""
        report: AuditReport | None = None
        for chunk in self.graph.stream(initial, config=self._config(), stream_mode="updates"):
            for node, raw_update in chunk.items():
                update = raw_update or {}
                yield ProgressEvent(
                    node=node,
                    label=NODE_LABELS.get(node, node),
                    trace=list(update.get("trace", [])),
                    errors=list(update.get("errors", [])),
                )
                if isinstance(update.get("report"), AuditReport):
                    report = update["report"]
        if report is None:
            raise AuditExecutionError("The graph finished without producing a report")
        yield report

    @validated
    def run(self, project_path: FilePath) -> AuditReport:
        """Run the full audit on a document file and return the report (CLI / scripts)."""
        return self.invoke(self._state_for_file(project_path))

    @validated
    def stream(self, project_path: FilePath) -> Iterator[ProgressEvent | AuditReport]:
        """Streamed variant of :meth:`run`."""
        return self.iter_events(self._state_for_file(project_path))

    def mermaid(self) -> str:
        return str(self.graph.get_graph().draw_mermaid())
