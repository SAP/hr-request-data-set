from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class TokenizedDataset(Dataset, ABC):
    """
    Base class to define datasets classes for finetuning

    Args:
        Dataset (_type_): _description_
        ABC (_type_): _description_
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, data_path: str, special_tokens: dict[str, str]):
        """

        Args:
            tokenizer (PreTrainedTokenizerBase):
            data_path (_type_): Folder path where files used for finetuning are saved
            special_tokens (_type_): tokens used for tokenization ( defined in conf file )
        """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_masks = []

        data: list[str] = self._load_dataset(data_path)

        max_length: int = max([len(tokenizer.encode(d)) for d in data])

        MAX_LENGTH_TOKENIZER: int = tokenizer.model_max_length
        if max_length > MAX_LENGTH_TOKENIZER:
            max_length = MAX_LENGTH_TOKENIZER

        bos: str = special_tokens["bos_token"]
        eos: str = special_tokens["eos_token"]

        for txt in data:
            gpt_text = f"{bos}{txt}{eos}"

            encoding_dicts = tokenizer(
                gpt_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

            self.input_ids.append(torch.tensor(encoding_dicts["input_ids"]))
            self.attention_masks.append(torch.tensor(encoding_dicts["attention_mask"]))

    @abstractmethod
    def _load_dataset(self, data_path: str) -> list[str]:
        """
        Read every file inside "data_path" and save them as a list of strings
        to be used for finetuning

        Args:
            data_path (str): relative path of data files

        Returns:
            list[str]: list of strings to be used for finetuning
        """

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_masks[idx],
        )
