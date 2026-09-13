from sqlalchemy.orm import Session

from app.database.models import Session as SessionModel, ModelRun, RetrievalRun, WebSearchRun, AuditEvent


class Repository:
    def __init__(self, session: Session):
        self.session = session

    def create_session(self, record: dict):
        model = SessionModel(**record)
        self.session.add(model)
        self.session.commit()
        return model

    def create_model_run(self, record: dict):
        model = ModelRun(**record)
        self.session.add(model)
        self.session.commit()
        return model

    def create_retrieval_run(self, record: dict):
        model = RetrievalRun(**record)
        self.session.add(model)
        self.session.commit()
        return model

    def create_web_search_run(self, record: dict):
        model = WebSearchRun(**record)
        self.session.add(model)
        self.session.commit()
        return model

    def create_audit_event(self, record: dict):
        model = AuditEvent(**record)
        self.session.add(model)
        self.session.commit()
        return model
