from __future__ import annotations

import logging

import hydra
import torch
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizerFast

from use_cases.ticket_classification.src import TicketClassifier, TicketDataset
from use_cases.ticket_classification.src.ticket_classifier import InterpretOutput, PredictOutput
from use_cases.ticket_classification.src.ticket_classifier_BERT import TicketClassifierBERT
from use_cases.ticket_classification.src.ticket_classifier_fasttext import TicketClassifierFastText
from util.util import load_survey_tickets, load_ticket_dataset

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_ticket_classifier_bert(
    model_name: str,
    use_gpu: bool,
    device: str,
    tickets_train: list[str],
    tickets_test: list[str],
    tickets_val: list[str],
    labels_train: list[int],
    labels_test: list[int],
    labels_val: list[int],
    training_args: dict,
) -> TicketClassifierBERT:
    # Calculate number of labels
    num_labels: int = len(set(labels_train + labels_test + labels_val))

    logger.info("TOKENIZING DATA...")

    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True, cache_dir="gen_cache")

    # convert our tokenized data into a torch Dataset
    train_dataset: TicketDataset = TicketDataset(tokenizer=tokenizer, tickets=tickets_train, labels=labels_train)
    val_dataset: TicketDataset = TicketDataset(tokenizer=tokenizer, tickets=tickets_val, labels=labels_val)
    test_dataset: TicketDataset = TicketDataset(tokenizer=tokenizer, tickets=tickets_test, labels=labels_test)

    logger.info("LOADING MODEL...")

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        cache_dir="gen_cache",
    )

    if use_gpu and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)

        model.to(device)

        model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    else:
        model.to("cpu")

    ticket_classifier = TicketClassifierBERT(
        tokenizer=tokenizer,
        model=model,
        train_data=train_dataset,
        valid_data=val_dataset,
        test_data=test_dataset,
        training_args=training_args,
        device=device,
    )

    return ticket_classifier


@hydra.main(config_path="conf/ticket_classification", config_name="config")
def main(cfg):
    logger.info("START")

    classificatin_model = cfg.classification_model
    assert classificatin_model in ["fasttext", "BERT"], "classificatin_model must be in ['fasttext', 'BERT']"

    use_gpu: bool = cfg.gpu.use_gpu
    device: str = cfg.gpu.device

    logger.info(f"USE_GPU: {use_gpu and torch.cuda.is_available()}")

    logger.info("LOADING DATA...")

    cfg_ticket_dataset: dict = cfg.ticket_dataset

    # Loading tickets from local folders, after creating them with ticket_generation
    tickets: list[str]
    labels: list[int]
    label_2_id: dict[str, int]
    tickets, _, labels, label_2_id, id_2_label = load_ticket_dataset(**cfg_ticket_dataset)

    train_size: float = cfg.train_size
    test_size: float = cfg.test_size
    validation_size: float = cfg.validation_size

    assert (
        train_size + test_size + validation_size == 1
    ), "The sum of train_size + test_size + validation_size must be equal to 1"

    # Define types of variables
    tickets_train: list[str]
    tickets_test: list[str]
    tickets_val: list[str]
    labels_train: list[int]
    labels_test: list[int]
    labels_val: list[int]

    # TRAIN-VAL SPLIT
    # Situation here
    # Val size = val_size
    # Train size = train_size + test_size
    tickets_train, tickets_val, labels_train, labels_val = train_test_split(
        tickets, labels, shuffle=True, test_size=validation_size, stratify=labels, random_state=42
    )

    # TRAIN-TEST SPLIT
    # Situation here
    # Test size = test_size
    # Train size = train_size
    # Validation size = validation_size
    # tickets_train, tickets_test, labels_train, labels_test = train_test_split(
    #     tickets_train,
    #     labels_train,
    #     shuffle=True,
    #     train_size=train_size / (train_size + test_size),
    #     stratify=labels_train,
    # )

    # Right now it load some tickets I wrote personally ( Gabriele ), to have some test tickets that
    # are not created by the model and do not trace back to the original templates, to see if the
    # model actually works ( of course I could be biased in writing them as I have also written the
    # templates, but it's still better than splitting the generated dataset )
    # The code for the classical split is the one commented above
    #
    # UPDATE: did a survey so now test tickets are taken from survey,
    #
    # tickets_test, labels_test = TicketDataset.load_handwritten_ticket(
    #     data_path="ticket_generation/data/", file_name="tickets_handwritten.json", label_2_id=label_2_id
    # )

    # Tickets taken from Excel files gathered from survey
    tickets_test, labels_test = load_survey_tickets(
        data_path="ticket_generation/data/survey_tickets", label_2_id=label_2_id
    )

    if classificatin_model == "BERT":

        model_name: str = cfg.model_name
        training_args: dict = cfg.training_args_bert

        logging.info(f"TRAINING ARGS: {training_args}")

        ticket_classifier = get_ticket_classifier_bert(
            model_name=model_name,
            use_gpu=use_gpu,
            device=device,
            tickets_train=tickets_train,
            tickets_val=tickets_val,
            tickets_test=tickets_test,
            labels_train=labels_train,
            labels_val=labels_val,
            labels_test=labels_test,
            training_args=training_args,
        )
    elif classificatin_model == "fasttext":
        training_args: dict = cfg.training_args_fasttext

        ticket_classifier = TicketClassifierFastText(
            tickets_train=tickets_train,
            tickets_val=tickets_val,
            tickets_test=tickets_test,
            labels_train=labels_train,
            labels_val=labels_val,
            labels_test=labels_test,
            training_args=training_args,
        )

    logger.info("TRAINING...")

    ticket_classifier.train()

    logger.info("EVALUATION...")

    metrics_evaluation = ticket_classifier.evaluate()
    logger.info(metrics_evaluation)

    logger.info("TEST...")

    execute_explanation: bool = cfg.explanation.execute

    if execute_explanation and classificatin_model == "BERT":
        logger.info("TEST WITH INTERPRETATION...")

        # In results there will be not only the classification result, but also the top words
        # which triggered the result
        results: list[InterpretOutput] = ticket_classifier.interpret_prediction(
            tickets_test, labels_test, top_n=cfg.explanation.top_n
        )
    else:
        results: PredictOutput = ticket_classifier.predict()
        result_metrics = results["metrics"]

        logger.info(f"label_2_id: \n{label_2_id}")

        logger.info(f"RESULT PREDICTION ON TEST SET:")
        for key, value in result_metrics.items():
            logger.info(f"{key}: {value}")

        display_predicted_wrong: bool = cfg.display_predicted_wrong

        if display_predicted_wrong:
            logger.info(f"TICKETS PREDICTED WRONG:")
            wrong_predicted_tickets: list[dict[str, str]] = TicketClassifier.get_wrong_tickets(
                tickets=tickets_test,
                pred_labels=results["predictions"],
                true_labels=results["true_labels"],
                id_2_label=id_2_label,
            )
            for i, _ticket in enumerate(wrong_predicted_tickets):
                logger.info(f"{i:>5} WRONG TICKET: {_ticket}")

    output_path: str = cfg.output_path
    TicketClassifier.save_results(results=results, output_path=output_path)


if __name__ == "__main__":
    main()
