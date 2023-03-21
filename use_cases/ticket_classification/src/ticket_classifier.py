from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, NamedTuple, Union


class InterpretOutput(NamedTuple):
    sentence: str
    top_n_words: list[tuple[str, float]]
    prediction: int
    true_label: int


class PredictOutput(NamedTuple):
    predictions: list[int]
    true_labels: list[int]
    metrics: dict[str, Any]


class TicketClassifier(ABC):
    """
    Class for classifying tickets based on their text to one of the given categories
    """

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self) -> PredictOutput:
        pass

    @staticmethod
    def get_wrong_tickets(
        tickets: list[str], pred_labels: list[int], true_labels: list[int], id_2_label: dict[int, str]
    ) -> list[dict[str, str]]:
        """
        Return the list of the wrongly classified tickets

        Args:
            tickets (list[str]): list of all test tickets
            pred_labels (list[int]): predicted labels
            true_labels (list[int]): true ;labels

        Return:
            list[tuple[str, int, int]]):
        """
        wrong_predicted_tickets: list[str] = []
        for ticket, pred_label, true_label in zip(tickets, pred_labels, true_labels):
            if pred_label != true_label:
                wrong_predicted_tickets.append(
                    {"ticket": ticket, "predicted": id_2_label[pred_label], "true": id_2_label[true_label]}
                )

        return wrong_predicted_tickets

    @staticmethod
    def save_results(results: Union[list[InterpretOutput], PredictOutput], output_path: str):
        """
        Save results to file

        Args:
            results (list[dict]): list of results
            output_path (str): path where the results are saved
        """

        _time: str = datetime.today().strftime("%Y_%m_%d_%H_%M")
        output_path_exist: bool = os.path.exists(output_path)

        # If output folder does not exist I create it
        if not output_path_exist:
            os.makedirs(output_path)

        with open(f"{output_path}/results_{_time}.json", "w") as f:
            json.dump(results, f)
