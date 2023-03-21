import numpy as np
import pandas as pd

from .employee_generator import EmployeeGenerator
from .personal_information_faker import PersonalInformationFaker


class EmployeeGenderPayGapGenerator(EmployeeGenerator):
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
            df_gender_pay_gap: pd.DataFrame = self.generate_gender_pay_gap_df(data=data, size=size)
            df: pd.DataFrame = pd.concat([df_pi, df_gender_pay_gap], axis=1)
            return df
        except ValueError as error:
            raise ValueError("Eror") from error

    def generate_gender_pay_gap_df(self, data: pd.DataFrame, size: int) -> pd.DataFrame:
        """
        Args:
            data (pd.DataFrame): dataset with various hourly percentual differences of wage between
                                 women and men ( columns: ['DiffMeanHourlyPercent'])
            size (int): number of data we want to create

        Returns:
            pd.DataFrame: returns a dataset of women workers with hourly difference of wage in percentual
        """

        # Sample records with replacement based on total employed ( occupations
        # with more employed people will be sampled more )
        data = data.sample(n=size, replace=True)

        # Create previous salary as: Normal(mean=DiffMeanHourlyPercent, std)
        data["DiffMeanHourlyPercent"] = np.random.normal(loc=data["DiffMeanHourlyPercent"], scale=0.2)

        # Round the difference of hourly wage to the first decimal
        data["DiffMeanHourlyPercent"] = data["DiffMeanHourlyPercent"].round(1)

        # Reset index to concat later
        data = data.reset_index()

        return data
