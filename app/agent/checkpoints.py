from typing import Any


class CheckpointStore:
    """Simple in-memory checkpoint abstraction for LangGraph-style persisted state snapshots."""

    def __init__(self):
        self.snapshots: dict[str, dict[str, Any]] = {}

    def save(self, session_id: str, state: dict[str, Any]) -> None:
        self.snapshots[session_id] = state

    def load(self, session_id: str) -> dict[str, Any]:
        return self.snapshots.get(session_id, {})
