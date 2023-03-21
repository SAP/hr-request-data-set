from __future__ import annotations

import math

import pandas as pd

from .df_provider import DfProvider


class AbsenteeismDfProvider(DfProvider):
    """
    Class that provides the dataset of absences of employees, with the reason why they were absent and the
    time they were absent

    Args:
    columns (list[str], optional): Columns you want to extract from the dataset.
                                   Defaults to ["Reason for absence", "Month of absence", "Absenteeism time in hours"].
    number_of_data (int, optional): Number of rows you want to save from the dataset. Defaults to 200.
    shuffle (bool, optional): True if you want to shuffle the dataset before trimming the first number_of_data rows.
                              Defaults to False.
    dataset_path (str, optional): Path of the dataset. Defaults to "data".
    file_name (str): name of the file with extension you want to get your data from
    """

    def __init__(
        self,
        number_of_data: int,
        shuffle: bool,
        dataset_path: str,
        file_name: str,
        columns: list[str] = ["Reason for absence", "Month of absence", "Absenteeism time in hours"],
    ):
        super().__init__(
            columns=columns,
            number_of_data=number_of_data,
            shuffle=shuffle,
            dataset_path=dataset_path,
            file_name=file_name,
        )

    def get_dataframe(self) -> pd.DataFrame:
        # Separator used in dataset: ';'
        df_absenteeism: pd.DataFrame = super().get_dataframe()

        df_absenteeism = self._preprocess_dataset(df_absenteeism)

        return df_absenteeism

    def _preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        in order this method:
        - transform hours of absence in days of absence ( 1 day of work equals to 8 hours )

        Args:
            df (pd.DataFrame): df to preprocess

        Returns:
            pd.DataFrame: preprocessed dataset
        """

        def _get_time_in_days(hours: int) -> int:
            _days = math.ceil(hours / 8)

            if _days <= 0:
                _days = 1

            return _days

        df["Absenteeism_time_in_days"] = df["Absenteeism_time_in_hours"].apply(_get_time_in_days)

        df = df.drop(columns=["Absenteeism_time_in_hours"])

        # Drop rows where reason for absence is unknown
        df = df.drop(df[df["Reason_for_absence"] == 0].index)

        return df
