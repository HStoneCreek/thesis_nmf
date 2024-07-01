from abc import abstractmethod
from typing import Optional

from sklearn.utils import shuffle
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from genolytics.benchmark.tables.table_abstract import AbstractTable
from genolytics.models.abstract import AbstractModel



class AbstractBenchmark:

    def __init__(self,
                 model: AbstractModel):
        self.model = model

        self.engine = self.set_up_db()
        self.session = sessionmaker(bind=self.engine)()

    def set_up_db(self):

        engine = create_engine('sqlite:///results.db')

        AbstractTable.metadata.create_all(engine)

        return engine

    def evaluate(self, dropout: Optional[float] = None, *args, **kwargs):

        if dropout is not None:
            rows_to_drop = int(len(self.model.X) * dropout)
            self.model.X = shuffle(self.model.X, random_state=self.model.seed)[rows_to_drop:]
            if self.model.y is not None:
                self.model.y = shuffle(self.model.y, random_state=self.model.seed)[rows_to_drop:]

        self.model.fit(*args, **kwargs)

        self.model.evaluate()

        self.write_results(dropout=dropout)

    @abstractmethod
    def write_results(self,
                      dropout: float):
        raise NotImplementedError(f"Must be implemented by child.")


if __name__ == "__main__":

    pass



