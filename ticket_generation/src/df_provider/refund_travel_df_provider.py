from __future__ import annotations

import pandas as pd

from .df_provider import DfProvider


class RefundTravelDfProvider(DfProvider):
    """
    Class that provides the dataset of complaints

    Args:
    columns (list[str], optional): Columns you want to extract from the dataset.
                                   Defaults to ["Name_source","City_source","Country_source","Name_dest",
                                                "City_dest","Country_dest"]
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
        df_complaint: pd.DataFrame = super().get_dataframe()

        df_complaint = self._preprocess_dataset(df_complaint)

        return df_complaint

    def _preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        in order this method:
        - Rename columns

        Args:
            df (pd.DataFrame): df to preprocess

        Returns:
            pd.DataFrame: preprocessed dataset
        """

        df = df.rename(
            columns={
                "Name_source": "airport_from",
                "City_source": "from",
                "Name_dest": "airport_to",
                "City_dest": "to",
            }
        )

        return df
