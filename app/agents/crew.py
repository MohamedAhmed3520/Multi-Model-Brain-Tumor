from crewai import Crew, Process

from app.agents.agents import (
    AUDIO_AGENT,
    EVIDENCE_AGENT,
    RAG_AGENT,
    REPORT_AGENT,
    VISION_AGENT,
    WEB_AGENT,
)


class CrewAIOrchestrator:
    """Real CrewAI orchestration facade that builds a Crew with a sequential Process."""

    def __init__(self):
        self.agents = [
            VISION_AGENT,
            AUDIO_AGENT,
            RAG_AGENT,
            WEB_AGENT,
            EVIDENCE_AGENT,
            REPORT_AGENT,
        ]
        self.crew = Crew(
            agents=self.agents,
            tasks=[],
            process=Process.sequential,
            verbose=False,
        )

    def run(self, workflow: str):
        return {
            'workflow': workflow,
            'agents': [agent.name for agent in self.agents],
            'status': 'planned',
            'crew_class': self.crew.__class__.__name__,
            'process': str(self.crew.process),
        }
