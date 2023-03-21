from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DfProvider(ABC):
    """
    Class that provides the dataset needed

        Args:
            columns (list[str]): list of columns you want to select
            file_name (str): name of the file with extension you want to get your data from
            number_of_data (int, optional): number of data you want to select (if <= 0, select all). Defaults to 200.
            shuffle (bool, optional): True if you want to shuffle the data. Defaults to False.
            dataset_path (str, optional): Path to find the dataset file. Defaults to "data".
    """

    def __init__(
        self,
        columns: list[str],
        file_name: str,
        number_of_data: int = 200,
        shuffle: bool = False,
        dataset_path: str = "data",
    ):
        self.columns = columns
        self.number_of_data = number_of_data
        self.shuffle = shuffle
        self.dataset_path = dataset_path
        self.file_name = file_name

    @abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_csv(filepath_or_buffer=f"{self.dataset_path}/{self.file_name}")

        df = df[self.columns]

        if self.shuffle:
            df = df.sample(frac=1)

        if self.number_of_data >= 1:
            df = df.iloc[: self.number_of_data]

        return df

    @abstractmethod
    def _preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
