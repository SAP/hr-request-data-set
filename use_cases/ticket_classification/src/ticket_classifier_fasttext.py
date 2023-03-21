from __future__ import annotations

import fasttext
from sklearn.metrics import accuracy_score, confusion_matrix

from use_cases.ticket_classification.src.ticket_classifier import PredictOutput, TicketClassifier


class TicketClassifierFastText(TicketClassifier):
    """
    Class for classifying tickets based on their text to one of the given categories
    """

    def __init__(
        self,
        tickets_train: list[str],
        tickets_test: list[str],
        tickets_val: list[str],
        labels_train: list[int],
        labels_test: list[int],
        labels_val: list[int],
        training_args: dict,
    ):
        self.tickets_train = tickets_train
        self.tickets_test = tickets_test
        self.tickets_val = tickets_val
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.labels_val = labels_val
        self.training_args = training_args

        self.model = None
        self.file_train = "use_cases/ticket_classification/data/tickets.train"
        self.file_val = "use_cases/ticket_classification/data/tickets.val"
        self.file_test = "use_cases/ticket_classification/data/tickets.test"

        self.prepare_data_files()

    def prepare_data_files(self):

        # https://fasttext.cc/docs/en/supervised-tutorial.html#getting-and-preparing-the-data
        # Format that fasttext need:
        # One line for record
        # Line: __label__{label} {text_data}
        #
        # Example:
        # __label__negative I hate this  !

        def write_data_to_file(tickets: list[str], labels: list[int], file_path: str):
            lines: list[str] = []
            for ticket, label in zip(tickets, labels):
                lines.append(f"__label__{label} {ticket}\n")

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

        write_data_to_file(self.tickets_train, self.labels_train, self.file_train)
        write_data_to_file(self.tickets_val, self.labels_val, self.file_val)
        write_data_to_file(self.tickets_test, self.labels_test, self.file_test)

    def train(self):
        self.model = fasttext.train_supervised(input=self.file_train, **self.training_args)

        self.model.save_model("use_cases/ticket_classification/model/model_fasttext.bin")

    def evaluate(self):
        precision: float
        recall: float

        # metrics: nexamples, precision(), recall()
        _, precision, recall = self.model.test(self.file_val)

        f1_score = (2 * precision * recall) / (precision + recall)

        metrics = {"precision": precision, "recall": recall, "f1_score": f1_score}

        return metrics

    def predict(self) -> PredictOutput:

        # Fasttext does not let newlines for predictions
        # https://github.com/facebookresearch/fastText/issues/1079#issuecomment-637440314
        def remove_newlines(ticket: str) -> str:
            return ticket.replace("\n", " ")

        tickets_test_wo_newlines: list[str] = list(map(remove_newlines, self.tickets_test))

        predictions: list[list[str]] = self.model.predict(tickets_test_wo_newlines, k=1)[0]

        def format_prediction(pred: list[str]) -> int:
            return int(pred[0][-1])

        predictions_labels: list[int] = list(map(format_prediction, predictions))

        accuracy: float = accuracy_score(self.labels_test, predictions_labels)

        precision: float
        recall: float

        # metrics: nexamples, precision(), recall()
        _, precision, recall = self.model.test(self.file_test)

        f1_score = (2 * precision * recall) / (precision + recall)

        conf_matrix: list[list[int]] = confusion_matrix(self.labels_test, predictions_labels).tolist()

        metrics = {"test_f1": f1_score, "test_accuracy": accuracy, "test_confusion_matrix": conf_matrix}

        results: PredictOutput = {
            "predictions": predictions_labels,
            "true_labels": self.labels_test,
            "metrics": metrics,
        }

        return results
