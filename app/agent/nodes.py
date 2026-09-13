from typing import Any


class NodeResult:
    def __init__(self, node: str, status: str, payload: dict[str, Any] | None = None):
        self.node = node
        self.status = status
        self.payload = payload or {}


class MedicalGraphNodes:
    """Stateful LangGraph node callables that update the shared MedicalAgentState dict."""

    def __init__(self):
        self.nodes = {
            'vision': self.vision_node,
            'audio': self.audio_node,
            'rag': self.rag_node,
            'web': self.web_node,
            'validate': self.validate_node,
            'report': self.report_node,
        }

    def vision_node(self, state: dict[str, Any]) -> dict[str, Any]:
        findings = dict(state.get('agent_findings') or {})
        findings['vision'] = {
            'classification': state.get('classification'),
            'detection': state.get('detection'),
            'segmentation': state.get('segmentation'),
        }
        return {'agent_findings': findings}

    def audio_node(self, state: dict[str, Any]) -> dict[str, Any]:
        findings = dict(state.get('agent_findings') or {})
        findings['audio'] = {'transcription': state.get('transcription')}
        return {'agent_findings': findings}

    def rag_node(self, state: dict[str, Any]) -> dict[str, Any]:
        findings = dict(state.get('agent_findings') or {})
        findings['rag'] = {'retrieved_evidence': state.get('retrieved_evidence', [])}
        return {'agent_findings': findings}

    def web_node(self, state: dict[str, Any]) -> dict[str, Any]:
        findings = dict(state.get('agent_findings') or {})
        findings['web'] = {'web_evidence': state.get('web_evidence', [])}
        return {'agent_findings': findings}

    def validate_node(self, state: dict[str, Any]) -> dict[str, Any]:
        findings = dict(state.get('agent_findings') or {})
        findings['validate'] = {
            'status': 'validated',
            'errors': state.get('errors', []),
            'evidence_bundle': state.get('evidence_bundle'),
        }
        return {'agent_findings': findings}

    def report_node(self, state: dict[str, Any]) -> dict[str, Any]:
        findings = dict(state.get('agent_findings') or {})
        findings['report'] = {'final_response': state.get('final_response')}
        return {'agent_findings': findings}

    def run(self, node: str, state: dict[str, Any]):
        return self.nodes[node](state)
