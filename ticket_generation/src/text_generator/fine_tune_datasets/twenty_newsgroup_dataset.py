from sklearn.datasets import fetch_20newsgroups
from transformers import GPT2TokenizerFast

from .tokenized_dataset import TokenizedDataset


class TwentyNewsGroupDataset(TokenizedDataset):
    def __init__(self, tokenizer: GPT2TokenizerFast, data_path, special_tokens):
        super().__init__(tokenizer, data_path, special_tokens)

    def _load_dataset(self, data_path: str):

        docs = fetch_20newsgroups(data_home=data_path, subset="all", remove=("headers", "footers", "quotes"))

        data = docs["data"]

        return data
