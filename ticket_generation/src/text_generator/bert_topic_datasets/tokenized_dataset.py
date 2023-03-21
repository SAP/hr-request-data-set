from __future__ import annotations

from abc import ABC, abstractmethod

from gensim.utils import simple_preprocess
from spacy.lang.en.stop_words import STOP_WORDS
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase


class TokenizedDataset(Dataset, ABC):
    """
    Base class to define datasets classes for finetuning

    Args:
        Dataset (_type_): _description_
        ABC (_type_): _description_
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, data_path: str):
        """

        Args:
            tokenizer (PreTrainedTokenizerBase):
            data_path (_type_): Folder path where files used for finetuning are saved
            special_tokens (_type_): tokens used for tokenization ( defined in conf file )
        """
        self.tokenizer = tokenizer

        data: list[str] = self._load_dataset(data_path)

        stop_words = list(STOP_WORDS)

        self.text_tokens: list[list[str]] = []

        for txt in tqdm(data):
            txt_wo_stopwords = " ".join(
                [token for token in simple_preprocess(str(txt), deacc=True) if token not in stop_words]
            )
            tokenized_text: list[str] = tokenizer.tokenize(txt_wo_stopwords)

            self.text_tokens.append(tokenized_text)

    def get_text_tokens(self) -> list[list[str]]:
        return self.text_tokens

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
        raise NotImplementedError
