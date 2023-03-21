from __future__ import annotations

import re

from sklearn.datasets import fetch_20newsgroups
from transformers import AutoTokenizer

from .tokenized_dataset import TokenizedDataset


class TwentyNewsGroupDataset(TokenizedDataset):
    def __init__(self, tokenizer: AutoTokenizer, data_path: str):
        super().__init__(tokenizer, data_path)

    def _load_dataset(self, data_path: str) -> list[str]:

        docs = fetch_20newsgroups(data_home=data_path, subset="all", remove=("headers", "footers", "quotes"))

        data: list[str] = docs["data"]

        data = [re.sub("\S*@\S*\s?", "", sent) for sent in data]
        # Remove new line characters
        data = [re.sub("\s+", " ", sent) for sent in data]
        # Remove distracting single quotes
        data = [re.sub("'", "", sent) for sent in data]

        return data
