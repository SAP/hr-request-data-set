import numpy as np
import pandas as pd

from .employee_generator import EmployeeGenerator
from .personal_information_faker import PersonalInformationFaker


class EmployeeSalaryGenerator(EmployeeGenerator):
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
            df_salary: pd.DataFrame = self.generate_salary_df(data=data, size=size)
            df: pd.DataFrame = pd.concat([df_pi, df_salary], axis=1)
            return df
        except ValueError as error:
            raise ValueError("Eror") from error

    def generate_salary_df(
        self, data: pd.DataFrame, size: int, range_increase_low: int = 5, range_increase_high: int = 10
    ) -> pd.DataFrame:
        """
        Args:
            data (pd.DataFrame): dataset with various work types and the avg. salary
                                 for the work
            size (int): number of data we want to create

        Returns:
            pd.DataFrame: returns a dataset of workers with their old salary ( created with a Normal
            distribution based on their job), their new requested salary ( based on a random increase of the
            old salary) and the increase
        """

        # Sample records with replacement based on total employed ( occupations
        # with more employed people will be sampled more )
        data = data.sample(n=size, weights="TOT_EMP", replace=True)

        data = data.rename(columns={"A_MEAN": "prev_salary", "OCC_TITLE": "work_title"})

        # Create previous salary as: Normal(mean=previous_salary, std)
        data["standard_error"] = 2 * (data["MEAN_PRSE"] * data["prev_salary"] / 100)
        data["prev_salary"] = np.random.normal(loc=data["prev_salary"], scale=data["standard_error"])

        # Round the salary to the hundreds ( ex. 37,123 -> 37,100 )
        data["prev_salary"] = data["prev_salary"].round(-2)

        # Create new salary requested as: prev_salary + increase * prev_salary
        _random_increase_of_salary: np.ndarray = np.random.randint(
            low=range_increase_low, high=range_increase_high, size=data.shape[0]
        )
        data["increase_in_percentage"] = _random_increase_of_salary / 100
        data["new_salary"] = data["prev_salary"] + data["prev_salary"] * data["increase_in_percentage"]

        # Round the salary to the hundreds ( ex. 37,123 -> 37,100 )
        data["new_salary"] = data["new_salary"].round(-2)

        # Transform salaries into integers
        data["prev_salary"] = data["prev_salary"].astype(int)
        data["new_salary"] = data["new_salary"].astype(int)

        # Transform increase of salary into text
        # Ex. 0.06 -> "6%"
        def _get_increase_in_percentage_text(increase: np.float32):
            _increase_rounded = int(increase * 100)
            _increase_text = f"{_increase_rounded}%"

            return _increase_text

        data["increase_in_percentage"] = data["increase_in_percentage"].apply(_get_increase_in_percentage_text)

        # drop unwanted columns from data
        data = data.drop(columns=["TOT_EMP", "MEAN_PRSE"])

        # Reset index to concat later
        data = data.reset_index()

        return data
