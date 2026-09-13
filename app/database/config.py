import os
from dataclasses import dataclass


@dataclass
class DatabaseSettings:
    database_url: str = os.getenv('DATABASE_URL', 'sqlite:///./data/medical_ai.db')


settings = DatabaseSettings()
