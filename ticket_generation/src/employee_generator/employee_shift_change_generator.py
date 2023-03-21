from __future__ import annotations

from random import choices

import pandas as pd

from .employee_generator import EmployeeGenerator
from .personal_information_faker import PersonalInformationFaker


class EmployeeShiftChangeGenerator(EmployeeGenerator):
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
            df_shift_change: pd.DataFrame = self.generate_shift_change_df(data=data, size=size)
            df: pd.DataFrame = pd.concat([df_pi, df_shift_change], axis=1)
            return df
        except ValueError as error:
            raise ValueError("Eror") from error

    def generate_shift_change_df(
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

        # TODO: Possible shifts hard-coded here, to move later
        work_shifts = [("06:00", "14:00"), ("14:00", "22:00"), ("22:00", "06:00")]

        # Sample with replacement
        work_shifts_old: list[tuple[str, str]] = choices(work_shifts, k=size)
        work_shifts_new: list[tuple[str, str]] = choices(work_shifts, k=size)

        # I pick only starting time
        work_shifts_old_start: list[str] = list(map(lambda shift: shift[0], work_shifts_old))
        work_shifts_new_start: list[str] = list(map(lambda shift: shift[0], work_shifts_new))

        # Sample records with replacement
        data = data.sample(n=size, replace=True)

        data = pd.DataFrame(
            data={
                "reason_of_change": data["reason_of_change"],
                "old_work_shift": work_shifts_old_start,
                "new_work_shift": work_shifts_new_start,
            }
        )

        # Reset index to concat later
        data = data.reset_index()

        return data
