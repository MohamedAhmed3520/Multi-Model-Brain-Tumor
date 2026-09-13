from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.checkpoints import CheckpointStore
from app.agent.nodes import MedicalGraphNodes
from app.agent.router import Router
from app.agent.state import MedicalAgentState


class MedicalGraph:
    """LangGraph StateGraph-backed workflow controller for the multimodal medical RAG pipeline."""

    def __init__(self):
        self.nodes = MedicalGraphNodes()
        self.router = Router()
        self.checkpoints = CheckpointStore()
        self.graph = None
        self.compiled = None

    def build_state_graph(self):
        graph = StateGraph(MedicalAgentState)
        for node_name in self.nodes.nodes:
            graph.add_node(node_name, self.nodes.nodes[node_name])

        # Start routing from the patient request or attached files.
        graph.add_conditional_edges(
            START,
            lambda state: self.router.route(state),
            path_map={
                'vision': 'vision',
                'audio': 'audio',
                'rag': 'rag',
                'web': 'web',
            },
        )

        # A deterministic evidence review/report branch that honors the requested flow order.
        for node in ('vision', 'audio', 'rag', 'web'):
            graph.add_edge(node, 'validate')
        graph.add_edge('validate', 'report')
        graph.add_edge('report', END)

        return graph

    def compile(self):
        if self.compiled is None:
            self.graph = self.build_state_graph()
            self.compiled = self.graph.compile()
        return self.compiled

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        compiled = self.compile()
        result = compiled.invoke(state)
        session_id = state.get('session_id', 'default')
        self.checkpoints.save(session_id, result)
        return result
