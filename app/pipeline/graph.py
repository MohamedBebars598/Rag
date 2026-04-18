from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.pipeline.nodes.chunker import chunk_node
from app.pipeline.nodes.embedder import embed_node
from app.pipeline.nodes.extractor import extract_node
from app.pipeline.state import PipelineState


def build_pipeline() -> StateGraph:
    """Construct and compile the CV ingestion LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    graph.add_node("extract", extract_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("embed", embed_node)

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", END)

    return graph.compile()


pipeline = build_pipeline()
