from __future__ import annotations

import pandas as pd

from .df_provider import DfProvider


class ComplaintDfProvider(DfProvider):
    """
    Class that provides the dataset of complaints

    Args:
    columns (list[str], optional): Columns you want to extract from the dataset. Defaults to ["to_who","complaint"]
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
        number_of_data: int = -1,
        columns: list[str] = ["to_who", "complaint"],
    ):
        super().__init__(
            columns=columns,
            number_of_data=number_of_data,
            shuffle=shuffle,
            dataset_path=dataset_path,
            file_name=file_name,
        )

    def get_dataframe(self) -> pd.DataFrame:
        df_complaint: pd.DataFrame = super().get_dataframe()

        df_complaint = self._preprocess_dataset(df_complaint)

        return df_complaint

    def _preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        in order this method:
        - No preprocessing for this class

        Args:
            df (pd.DataFrame): df to preprocess

        Returns:
            pd.DataFrame: preprocessed dataset
        """

        return df
