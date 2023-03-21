from __future__ import annotations

import pandas as pd

from .df_provider import DfProvider


class GenderPayGapDfProvider(DfProvider):
    """
    Class that provides the dataset of average difference in salaries between women and men

    Args:
    columns (list[str], optional): Columns you want to extract from the dataset.
                                   Defaults to ["DiffMeanHourlyPercent"]
                                   Set in Hydra config file
    number_of_data (int, optional): Number of rows you want to save from the dataset. Defaults to -1.
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
        df_gender_pay_gap: pd.DataFrame = super().get_dataframe()

        df_gender_pay_gap = self._preprocess_dataset(df_gender_pay_gap)

        return df_gender_pay_gap

    def _preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        in order this method:
        - Drop rows with NA values
        - Convert percent difference into a float
        - Remove rows where women are paid more than men

        Args:
            df (pd.DataFrame): df to preprocess ()

        Returns:
            pd.DataFrame: preprocessed dataset
        """

        df = df.dropna()

        df = df.astype({"DiffMeanHourlyPercent": "float32"})

        # Remove rows where women are paid more than men
        df = df[df["DiffMeanHourlyPercent"] > 0]

        return df
