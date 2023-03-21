from __future__ import annotations

from typing import NamedTuple, Optional, TypedDict, Union

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizerFast, EvalPrediction, Trainer, TrainingArguments
from transformers_interpret import SequenceClassificationExplainer

from use_cases.ticket_classification.src.ticket_classifier import InterpretOutput, PredictOutput, TicketClassifier
from use_cases.ticket_classification.src.ticket_dataset import TicketDataset


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, tuple[np.ndarray]]]
    metrics: Optional[dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: dict[str, float]


class MetricOutput(TypedDict):
    f1: np.ndarray
    accuracy: float
    confusion_matrix: np.ndarray


class TicketClassifierBERT(TicketClassifier):
    """
    Class for classifying tickets based on their text to one of the given categories
    """

    def __init__(
        self,
        tokenizer: BertTokenizerFast,
        model: BertForSequenceClassification,
        train_data: TicketDataset,
        valid_data: TicketDataset,
        test_data: TicketDataset,
        training_args: dict,
        device: str,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.training_args = training_args
        self.device = device

        training_args = TrainingArguments(**self.training_args)
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.valid_data,
            compute_metrics=self.compute_metrics,
        )

    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def multi_label_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> MetricOutput:
        """
        Calculate f1_score, accuracy and confuzion matrix for the predicted labels

        Args:
            predictions : predicted logits ( matrix NxC, with N:number of data and C:number of classes)
            labels : one hot encoded real labels of the N records
        """

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / np.sum(e_x, axis=1, keepdims=True)

        predictions_normalized: np.ndarray = softmax(predictions)

        y_pred: np.ndarray = np.argmax(predictions_normalized, axis=1)
        y_true: np.ndarray = np.argmax(labels, axis=1)

        accuracy: float = accuracy_score(y_true, y_pred)

        f1_micro_average: np.ndarray = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
        conf_matrix: np.ndarray = confusion_matrix(y_true, y_pred).tolist()

        metrics: MetricOutput = {"f1": f1_micro_average, "accuracy": accuracy, "confusion_matrix": conf_matrix}
        return metrics

    def compute_metrics(self, p: EvalPrediction) -> MetricOutput:
        preds: np.ndarray = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result: MetricOutput = self.multi_label_metrics(predictions=preds, labels=p.label_ids)
        return result

    def train(self):
        """
        Train the model

        Returns:
            result_train (TrainOutput): statistics of training
        """
        self.trainer.train()

        self.model.save_pretrained("use_cases/ticket_classification/model/model_BERT.bin")

    def evaluate(self):
        """
        Evaluate the model
        """
        metrics = self.trainer.evaluate()

        return metrics

    def predict(self) -> PredictOutput:
        """
        Classify labeled data ( Use label only to compute metrics )

        Args:

        Returns:
            results: predicted labels and metrics
        """
        predictions: PredictionOutput = self.trainer.predict(self.test_data)

        # Remove predictions logits
        results: PredictOutput = {
            "predictions": predictions[0].argmax(axis=1).tolist(),
            "true_labels": self.test_data.labels.argmax(axis=1).tolist(),
            "metrics": predictions[2],
        }

        return results

    def interpret_prediction(self, test_data: list[str], labels_test: list[int], top_n: int) -> list[InterpretOutput]:
        """
        As the predict method, this method also makes prediction. However, it uses the transform_interpret library
        to calculate also the top_n words used to make a decision on the predictions

        Args:
            test_data ( list[str] ): list of tickets text
            labels_test (list[int]): list of labels indexes
            top_n (int): Top n words

        Returns:
            _type_: _description_
        """
        cls_explainer = SequenceClassificationExplainer(self.model, self.tokenizer)

        results: list[InterpretOutput] = []
        for data, true_label in tqdm(zip(test_data, labels_test), total=len(test_data)):
            explanation: list[tuple[str, float]] = cls_explainer(data)
            top_n_words: list[str] = sorted(explanation, key=lambda tup: -tup[1])[:top_n]
            prediction: int = int(cls_explainer.predicted_class_index)

            results.append(
                {"sentence": data, "top_n_words": top_n_words, "prediction": prediction, "true_label": true_label}
            )

        return results
