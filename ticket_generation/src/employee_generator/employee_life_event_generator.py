import pandas as pd

from .employee_generator import EmployeeGenerator
from .personal_information_faker import PersonalInformationFaker


class EmployeeLifeEventGenerator(EmployeeGenerator):
    """ """

    def __init__(self, personal_info_faker: PersonalInformationFaker = PersonalInformationFaker()):
        self.personal_info_faker = personal_info_faker

    def generate_employees(self, data: pd.DataFrame, size: int = 1) -> pd.DataFrame:
        """
        Generates synthetic employees

        Args:
            size: number of employees to create.

        Returns:
            df: dataframe containing synthetic employees.

        Raises:
            ValueError:
        """
        try:
            df_pi: pd.DataFrame = self.personal_info_faker.generate_profiles(size=size)
            df_life_event: pd.DataFrame = self.generate_life_event_df(data=data, size=size)
            df: pd.DataFrame = pd.concat([df_pi, df_life_event], axis=1)
            return df
        except ValueError as error:
            raise ValueError("Error") from error

    def generate_life_event_df(
        self,
        data: pd.DataFrame,
        size: int,
    ) -> pd.DataFrame:
        """
        Args:
            data (pd.DataFrame):
            size (int): number of data we want to create

        Returns:
            pd.DataFrame:
        """

        # Sample records with replacement
        data = data.sample(n=size, replace=True)

        # Reset index to concat later
        data = data.reset_index()

        return data
