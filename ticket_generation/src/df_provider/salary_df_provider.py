from __future__ import annotations

import numpy as np
import pandas as pd

from .df_provider import DfProvider


class SalaryDfProvider(DfProvider):
    """
    Class that provides the dataset of salaries of employees, with their work title and their previous salary,
    the new requested increase of salary and the correspondant new salary requested

    Args:
    columns (list[str], optional): Columns you want to extract from the dataset.
                                   Defaults to  ["OCC_TITLE", "TOT_EMP", "A_MEAN", "MEAN_PRSE"].
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
        df_salaries: pd.DataFrame = super().get_dataframe()

        df_salaries = self._preprocess_dataset(df_salaries)

        return df_salaries

    def _preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        in order this method:
        - Remove first row (All occupancies), we don't need a summary row
        - Remove rows with NaN values ( represented by '*' character)
        - Remove thousands separators ',' from integers
        - Change type of columns to integers/float when needed

        Args:
            df (pd.DataFrame): df to preprocess ()

        Returns:
            pd.DataFrame: preprocessed dataset
        """

        df = df.iloc[1:, :]

        df = df.replace("*", np.NaN)
        df = df.dropna()

        df[["TOT_EMP", "A_MEAN"]] = (
            df[["TOT_EMP", "A_MEAN"]].astype("string").apply(lambda column: column.str.replace(",", ""))
        )

        df = df.astype({"TOT_EMP": "int32", "A_MEAN": "int32", "MEAN_PRSE": "float32"})

        return df
