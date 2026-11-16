"""Page 1 - Overview, architecture, regulatory calendar, service health."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.compliance.timeline import all_deadlines
from src.core.config import Settings
from src.rag.preprocess_laws import corpus_stats, preprocess_all, write_jsonl
from src.ui import services

ARCHITECTURE_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#eef3f8", color="#1f4e79",
        fontname="Helvetica", fontsize=10];
  edge [color="#1f4e79", fontname="Helvetica", fontsize=9];
  subgraph cluster_kb { label="Knowledge base (offline)"; style=dashed; color="#90a4ae";
    pdf [label="Official PDFs\\nAI Act (EN) · GDPR (FR)"];
    pre [label="Legal preprocessing\\narticle / chapter / annex\\n+ metadata & dates"];
    qd [label="Qdrant\\n(dense bge-m3)", shape=cylinder];
    bm [label="BM25\\n(lexical)", shape=cylinder];
    pdf -> pre -> qd; pre -> bm;
  }
  up [label="Project .pdf / .pptx", fillcolor="#fff3e0"];
  parse [label="Parser\\npymupdf4llm · python-pptx"];
  prof [label="Project profiling"];
  rw [label="Query Rewriter\\nAI Act / GDPR sub-questions"];
  ret [label="Hybrid Retrieval\\nRRF + Reranker"];
  rt [label="Rethinker\\n(critic, <= 2 loops)", fillcolor="#fce4ec"];
  ai [label="Agent\\nAI Act classification"];
  gd [label="Agent\\nGDPR audit"];
  sy [label="Agent\\nSynthesis & recommendations"];
  rep [label="Report · Checklist\\nPDF / JSON · Chatbot", fillcolor="#e8f5e9"];
  up -> parse -> prof -> rw -> ret -> rt;
  rt -> ret [label="insufficient", style=dashed];
  rt -> ai [label="sufficient"]; rt -> gd;
  ai -> sy; gd -> sy; sy -> rep;
  qd -> ret [style=dotted]; bm -> ret [style=dotted];
}
"""
REGULATION_COLORS = {"AI Act": "#1f4e79", "GDPR": "#2e7d32"}
MAX_LABEL_CHARS = 38


def _timeline(today: date) -> None:
    deadlines = all_deadlines()
    regs = st.multiselect("Regulations", ["AI Act", "GDPR"], default=["AI Act", "GDPR"])
    selected = [d for d in deadlines if d.regulation.short in regs]
    if not selected:
        st.info("Select at least one regulation.")
        return
    df = pd.DataFrame(
        [
            {
                "Date": d.date,
                "Regulation": d.regulation.short,
                "Deadline": d.title,
                "Legal basis": d.legal_basis,
                "Amended by": d.amended_by or "",
                "Status": "✅ Applicable" if d.is_past(today) else f"⏳ D-{d.days_remaining(today)}",
            }
            for d in selected
        ]
    )
    fig = go.Figure()
    for reg, group in df.groupby("Regulation"):
        fig.add_trace(
            go.Scatter(
                x=group["Date"],
                y=[str(reg)] * len(group),
                mode="markers+text",
                marker={"size": 16, "color": REGULATION_COLORS.get(str(reg), "#555"), "symbol": "diamond"},
                text=[
                    t if len(t) < MAX_LABEL_CHARS else t[: MAX_LABEL_CHARS - 2] + "…"
                    for t in group["Deadline"]
                ],
                textposition="top center",
                textfont={"size": 10},
                customdata=group[["Deadline", "Legal basis", "Status", "Amended by"]].to_numpy(),
                hovertemplate="<b>%{customdata[0]}</b><br>%{x|%d/%m/%Y}<br>%{customdata[1]}<br>"
                "%{customdata[2]}<br>%{customdata[3]}<extra></extra>",
                name=str(reg),
            )
        )
    fig.add_vline(x=today.isoformat(), line_dash="dash", line_color="#c62828")
    fig.add_annotation(x=today.isoformat(), y=1.08, yref="paper", text="Today", showarrow=False)
    fig.update_layout(
        height=330,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        xaxis={"title": None, "tickformat": "%m/%Y"},
        yaxis={"title": None},
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        df.assign(Date=lambda x: pd.to_datetime(x["Date"]).dt.strftime("%d/%m/%Y")),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Deadline details"):
        for d in selected:
            st.markdown(f"**{d.date:%d/%m/%Y} - {d.title}** ({d.legal_basis})  \n{d.description}")


def _health(settings: Settings) -> None:
    ollama = services.ollama_health()
    store = services.store_health()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Ollama (LLM & embeddings)**")
        if not ollama.reachable:
            st.error(f"Unreachable: {settings.ollama_base_url}")
        elif ollama.missing_models:
            st.warning(f"Missing models: {', '.join(ollama.missing_models)}")
        else:
            st.success("Operational")
        st.caption(
            f"LLM: `{settings.llm_model}` · judge: `{settings.judge_model}` · "
            f"embeddings: `{settings.embedding_model}`"
        )
    with c2:
        st.markdown("**Qdrant (vector database)**")
        if not store.reachable:
            st.error("Unreachable")
            st.caption(store.error or "")
        elif not store.collection_exists or store.points == 0:
            st.warning("Empty collection: run the indexing")
        else:
            st.success(f"{store.points} chunks indexed")
        st.caption(f"Collection: `{settings.qdrant_collection}`")
    with c3:
        st.markdown("**Reranker**")
        st.success(services.retriever().reranker.name)
        st.caption(f"Top-n after reranking: {settings.rerank_top_n}")


def _indexing(settings: Settings) -> None:
    recreate = st.checkbox("Recreate the collection (full re-indexing)", value=False)
    if not st.button("📚 Pre-index the legal texts", type="primary"):
        return
    with st.status("Indexing the legal corpus...", expanded=True) as status:
        st.write("Structure-aware splitting of the official PDFs (articles, chapters, annexes)...")
        chunks = preprocess_all(settings)
        write_jsonl(chunks, settings.processed_chunks_path)
        st.write(f"{len(chunks)} chunks produced.")
        st.write("Computing embeddings and indexing into Qdrant...")
        n = services.retriever().store.index_chunks(chunks, recreate=recreate)
        services.retriever.clear()
        services.auditor.clear()
        status.update(label=f"Indexing complete: {n} chunks", state="complete")
    st.json(corpus_stats(chunks).model_dump(), expanded=False)


def render() -> None:
    settings = services.settings()
    st.title("⚖️ EU AI Compliance Auditor")
    st.markdown(
        "Automated, **100% local** compliance audit of an AI project against the **EU AI Act** "
        '(Regulation (EU) 2024/1689, as amended by Regulation (EU) 2026/1744 "Digital Omnibus") '
        "and the **GDPR** (Regulation (EU) 2016/679). Upload your project deck: the agents "
        "identify the risk category, the applicable obligations and deadlines, audit the "
        "processing of personal data and produce a prioritised action plan."
    )
    tab_arch, tab_cal, tab_health = st.tabs(
        ["🏗️ Architecture", "📅 Regulatory calendar", "🩺 Health & indexing"]
    )
    with tab_arch:
        st.graphviz_chart(ARCHITECTURE_DOT, width="stretch")
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            "**Legal RAG**\n- Split by article/annex\n- Metadata: risk, date of application\n"
            "- Hybrid dense + BM25 (RRF)\n- Cross-encoder reranker"
        )
        c2.markdown(
            "**LangGraph agents**\n- Query rewriter\n- Rethinker (corrective loop <= 2)\n"
            "- AI Act classification\n- GDPR audit\n- Synthesis & action plan"
        )
        c3.markdown(
            "**Guardrails**\n- Deterministic deadlines\n- Citation validation\n"
            "- Pydantic structured outputs\n- Ragas evaluation in CI"
        )
    with tab_cal:
        _timeline(date.today())
    with tab_health:
        _health(settings)
        st.divider()
        _indexing(settings)


render()
