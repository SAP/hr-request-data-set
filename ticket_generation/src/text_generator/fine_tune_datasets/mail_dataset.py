from __future__ import annotations

import glob
import os

from tqdm import tqdm
from transformers import AutoTokenizer

from .tokenized_dataset import TokenizedDataset


class MailDataset(TokenizedDataset):
    def __init__(self, tokenizer: AutoTokenizer, data_path, special_tokens):
        super().__init__(tokenizer, data_path, special_tokens)

    def _load_dataset(self, data_path: str) -> list[str]:
        """
        Read every file inside "data_path" and save them as a list of strings
        to be used for finetuning

        Args:
            data_path (str): relative path of data files

        Returns:
            list[str]: list of strings to be used for finetuning
        """

        file_names: list[str] = [f for f in glob.glob(f"{data_path}/*")]
        file_names = [f for f in file_names if os.path.isfile(f)]
        data: list[str] = []

        for file_name in tqdm(file_names):
            with open(f"{file_name}") as f:
                data.append(f.read())

        return data
