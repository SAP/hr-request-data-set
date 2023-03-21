from __future__ import annotations

import json
from typing import Any, TypedDict

import torch
from transformers import PreTrainedTokenizerBase


class Ticket(TypedDict):
    ticket: str
    category: str
    sub_category: str


class SurveyTickets(TypedDict):
    age_range: str
    tickets: list[Ticket]


class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, tickets: list[str], labels: list[int]):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        max_length: int = max([len(tokenizer.encode(d)) for d in tickets])

        MAX_LENGTH_TOKENIZER: int = tokenizer.model_max_length
        if max_length > MAX_LENGTH_TOKENIZER:
            max_length = MAX_LENGTH_TOKENIZER

        for ticket_text, label in zip(tickets, labels):
            encoding_dicts = tokenizer(
                ticket_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

            self.input_ids.append(torch.tensor(encoding_dicts["input_ids"]))
            self.attention_masks.append(torch.tensor(encoding_dicts["attention_mask"]))
            self.labels.append(label)

        # BertForSequenceClassification needs one hot encoded labels
        # for multi-class classification
        # Ex. labels = [0, 2, 1] -> [[1, 0, 0],[0, 0, 1],[0, 1, 0]]
        self.labels = torch.nn.functional.one_hot(torch.tensor(labels)).float()

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }

        return item

    def __len__(self):
        return len(self.input_ids)

    # TODO: write in some other place/ better organization
    @staticmethod
    def load_handwritten_ticket(
        data_path: str, file_name: str, label_2_id: dict[str, int]
    ) -> tuple[list[str], list[int]]:
        """
        Load the file with ticket written manually by myself( Gabriele )
        File format:
        [
            {
                "id": 0,
                "ticket": "Hello, I would like to stay home the 2nd of August for a medical consultation. Is it possible?",
                "label": "Life event_Health issues"
            },
            ...
        ]
        The label is in the format `{Category}_{SubCategory}`


        Args:
            data_path (str): folder where the file is saved
            file_name (str): name of the file
            label_2_id (dict[str,int]): dictionary with correspondance label:id

        Returns:
            tuple[list[str], list[int]]: list of tickets text | list of tickets labels
        """
        with open(f"{data_path}/{file_name}") as f:
            tickets: list[dict[str, Any]] = json.load(f)

        tickets_texts: list[str] = list(map(lambda ticket: ticket["ticket"], tickets))
        tickets_labels: list[int] = list(map(lambda ticket: label_2_id[ticket["label"]], tickets))

        return tickets_texts, tickets_labels
