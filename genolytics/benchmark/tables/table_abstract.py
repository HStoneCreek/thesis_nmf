from sqlalchemy import Column, DateTime
from sqlalchemy.orm import declarative_base

from datetime import datetime


class AbstractTable(declarative_base()):

    __abstract__ = True

    # time at which the row was created/inserted into the database
    inserted_at = Column(DateTime, default=datetime.utcnow)
    # time at which the row was last changed/updated
    last_updated = Column(DateTime, default=datetime.utcnow)
