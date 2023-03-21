from __future__ import annotations

from collections import Counter, defaultdict
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from util import load_ticket_dataset


def get_entities_which_appear_together(tickets, entities, tickets_no_prompt, labels, id_2_label):
    total_count_ticket_with_more_entities_in_text = 0

    total_count_ticket_with_more_entities_in_text = defaultdict(lambda: 0)

    entities_appearing_together = defaultdict(lambda: Counter())

    for ticket, entity_list, ticket_no_prompt, label in tqdm(
        zip(tickets, entities, tickets_no_prompt, labels), total=len(tickets)
    ):

        index_start_ticket_text = ticket.find(ticket_no_prompt)

        set_entities_found = set()

        for entity in entity_list:
            if entity[0] > index_start_ticket_text:
                set_entities_found.add(entity[2])

        if len(set_entities_found) == 2:
            total_count_ticket_with_more_entities_in_text[id_2_label[label]] += 1

            entities_appearing_together[id_2_label[label]][tuple(set_entities_found)] += 1

    MINIMUM_COUNT = 10
    entity_tuple_filtered = set()
    for category in entities_appearing_together:
        for entity_tuple in entities_appearing_together[category]:
            if entities_appearing_together[category][entity_tuple] > MINIMUM_COUNT:
                entity_tuple_filtered.add(entity_tuple)

    pprint(total_count_ticket_with_more_entities_in_text)
    pprint(entities_appearing_together)

    values = [v for _, v in total_count_ticket_with_more_entities_in_text.items()]
    print(f"Total: {sum(values)}")

    return entity_tuple_filtered


def get_tickets_with_relations(tickets: list[str], entities, ticket_no_prompt, labels, entity_tuple_filtered):
    new_tickets = []

    ENT1_START = "[ENT1_START]"
    ENT1_END = "[ENT1_END]"
    ENT2_START = "[ENT2_START]"
    ENT2_END = "[ENT2_END]"

    for ticket, entity_list, ticket_no_prompt, label in tqdm(
        zip(tickets, entities, ticket_no_prompt, labels), total=len(tickets)
    ):

        index_start_ticket_text = ticket.find(ticket_no_prompt)

        entities_found = list()

        for entity in entity_list:
            if entity[0] > index_start_ticket_text:
                entities_found.append(entity)

        only_name_entities = [_entity[2] for _entity in entities_found]

        if tuple(only_name_entities) in entity_tuple_filtered and len(entities_found) == 2:

            start_first_entity = entities_found[0][0]
            end_first_entity = entities_found[0][1]
            start_second_entity = entities_found[1][0]
            end_second_entity = entities_found[1][1]

            assert start_first_entity < start_second_entity, "wrong order"

            first_part = ticket[:start_first_entity]
            head = ticket[start_first_entity:end_first_entity]
            second_part = ticket[end_first_entity:start_second_entity]
            target = ticket[start_second_entity:end_second_entity]
            third_part = ticket[end_second_entity:]
            text = (
                f"{first_part}{ENT1_START} {head} {ENT1_END}{second_part}{ENT2_START} {target} {ENT2_END}{third_part}"
            )

            relation = f"{only_name_entities[0]}_{only_name_entities[1]}"

            new_tickets.append({"text": text, "label": relation})

    return new_tickets


class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, tickets: list[str]):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        max_length: int = max([len(tokenizer.encode(d["text"])) for d in tickets])

        MAX_LENGTH_TOKENIZER: int = tokenizer.model_max_length
        if max_length > MAX_LENGTH_TOKENIZER:
            max_length = MAX_LENGTH_TOKENIZER

        for _ticket in tickets:

            ticket_text = _ticket["text"]
            label = _ticket["label"]

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
        self.labels = torch.nn.functional.one_hot(torch.tensor(self.labels)).float()

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }

        return item

    def __len__(self):
        return len(self.input_ids)


def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


def train(tickets_with_relations):
    labels_unique = list(set([_ticket["label"] for _ticket in tickets_with_relations]))

    # TRAIN MODEL ENTITY RELATION
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-large-cased", cache_dir="gen_cache", num_labels=len(labels_unique)
    )

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        model.to("cuda")

        model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    else:
        model.to("cpu")

    special_tokens = ["[ENT1_START]", "[ENT1_END]", "[ENT2_START]", "[ENT2_END]"]
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-cased",
        additional_special_tokens=special_tokens,
        cache_dir="gen_cache",
        use_fast=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    label_2_id: dict[str, int] = {label: idx for idx, label in enumerate(labels_unique)}
    # id_2_label: dict[int, str] = {idx: label for label, idx in label_2_id.items()}

    # Convert Ticket relation label to numerical id
    tickets_with_relations = [
        {"text": _ticket["text"], "label": label_2_id[_ticket["label"]]} for _ticket in tickets_with_relations
    ]

    tickets_with_relations_dataset = TicketDataset(tokenizer=tokenizer, tickets=tickets_with_relations)

    train_size = 0.8
    train_size = int(train_size * len(tickets_with_relations_dataset))
    val_size = len(tickets_with_relations_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(tickets_with_relations_dataset, [train_size, val_size])

    training_args = TrainingArguments(
        output_dir="util/random_scripts/data/model_bert_ner",  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=3,  # batch size per device during training
        per_device_eval_batch_size=3,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=1e-5,  # strength of weight decay
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="no",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results_evaluate = trainer.evaluate()

    pprint(results_evaluate)


def main():
    ticket_dataset = {
        "data_path": "ticket_generation/output",
        "template_path": "ticket_generation/data/templates",
        "template_file_name": "templates.json",
        "sample_size": 1,
        "remove_first_part": False,
        "remove_template_sections": False,
        "filter_tickets_by_file_name": "2022_11_27",  # If passed by command line and if it contains underscores it must be surrounded by '', like this: 'filter_tickets_by_file_name="2022_09_15"'
    }

    ticket_dataset_no_prompt = {
        "data_path": "ticket_generation/output",
        "template_path": "ticket_generation/data/templates",
        "template_file_name": "templates.json",
        "sample_size": 1,
        "remove_first_part": True,
        "remove_template_sections": False,
        "filter_tickets_by_file_name": "2022_11_27",  # If passed by command line and if it contains underscores it must be surrounded by '', like this: 'filter_tickets_by_file_name="2022_09_15"'
    }

    tickets, entities, labels, _, id_2_label = load_ticket_dataset(**ticket_dataset)
    tickets_no_prompt, _, _, _, _ = load_ticket_dataset(**ticket_dataset_no_prompt)

    # Find tickets with more than one entity and return the entities that appear together in
    # more than 10 tickets
    entity_tuple_filtered = get_entities_which_appear_together(
        tickets, entities, tickets_no_prompt, labels, id_2_label
    )

    # Find tickets with the entities couple found before (entity_tuple_filtered) and
    # return them with the relation saved as 'label'
    tickets_with_relations = get_tickets_with_relations(
        tickets, entities, tickets_no_prompt, labels, entity_tuple_filtered
    )

    # TODO: TO CANCEL
    # tickets_with_relations = tickets_with_relations[:10]

    print(f"{len(tickets_with_relations)=}")

    # TRAIN ENTITY RELATION CLASSIFICATION MODEL
    train(tickets_with_relations)


if __name__ == "__main__":
    main()
