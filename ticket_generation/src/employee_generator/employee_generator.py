from abc import ABC, abstractmethod

import pandas as pd


class EmployeeGenerator(ABC):
    @abstractmethod
    def generate_employees(self, **kwargs) -> pd.DataFrame:
        pass
