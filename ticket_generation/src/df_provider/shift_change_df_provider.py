from __future__ import annotations

import pandas as pd

from .df_provider import DfProvider


class ShiftChangeDfProvider(DfProvider):
    """
    Class that provides the dataset of shift changes ()

    Args:
    columns (list[str], optional): Columns you want to extract from the dataset.
                                   Defaults to [Event,Description]
                                   Set in Hydra config file
    number_of_data (int, optional): Number of rows you want to save from the dataset. Defaults to -1 (all).
    shuffle (bool, optional): True if you want to shuffle the dataset. Defaults to False.
    dataset_path (str, optional): Path of the dataset. Defaults to "data".
    file_name (str): name of the file with extension you want to get your data from
    """

    def __init__(
        self,
        shuffle: bool,
        dataset_path: str,
        file_name: str,
        columns: list[str],
        number_of_data: int = -1,
    ):
        super().__init__(
            columns=columns,
            number_of_data=number_of_data,
            shuffle=shuffle,
            dataset_path=dataset_path,
            file_name=file_name,
        )

    def get_dataframe(self) -> pd.DataFrame:
        df_shift_change: pd.DataFrame = super().get_dataframe()

        df_shift_change = self._preprocess_dataset(df_shift_change)

        return df_shift_change

    def _preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        in order this method:
        - Rename columns of dataset

        Args:
            df (pd.DataFrame): df to preprocess

        Returns:
            pd.DataFrame: preprocessed dataset
        """

        df = df.rename(columns={"Description": "reason_of_change"})

        return df
