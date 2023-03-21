import json

import spacy
from tqdm import tqdm

from util import load_ticket_dataset


def save_survey_spacy_entities(NER):
    data_path = "ticket_generation/data/survey_tickets"

    with open(f"{data_path}/data_new.json") as f:
        tickets_survey = json.load(f)

    output = []

    for ticket in tqdm(tickets_survey):
        ticket_text = ticket["ticket"]
        entities = ticket["entities"]
        label = ticket["label"]
        id = ticket["id"]

        text = NER(ticket_text)

        entities_spacy = []
        for word in text.ents:
            entities_spacy.append([word.text, word.label_, word.start_char, word.end_char])

        entity_list = list(map(lambda e: [ticket_text[e[0] : e[1]], e[2], e[0], e[1]], entities))

        output.append(
            {
                "id": id,
                "ticket": ticket_text,
                "entities_spacy": entities_spacy,
                "entities_handwritten": entity_list,
                "label": label,
            }
        )

    output = sorted(output, key=lambda o: o["label"])

    with open("util/random_scripts/data/ner_survey_tickets.json", "w") as f:
        json.dump(output, f)

    print("SAVED NER survey")


def save_train_spacy_entities(NER):

    ticket_dataset = {
        "data_path": "ticket_generation/output",
        "template_path": "ticket_generation/data/templates",
        "template_file_name": "templates.json",
        "sample_size": 1,
        "remove_first_part": False,
        "remove_template_sections": False,
        "filter_tickets_by_file_name": "2022_11_16",  # If passed by command line and if it contains underscores it must be surrounded by '', like this: 'filter_tickets_by_file_name="2022_09_15"'
    }

    tickets_no_prompt_dataset = {
        "data_path": "ticket_generation/output",
        "template_path": "ticket_generation/data/templates",
        "template_file_name": "templates.json",
        "sample_size": 1,
        "remove_first_part": True,
        "remove_template_sections": False,
        "filter_tickets_by_file_name": "2022_11_16",  # If passed by command line and if it contains underscores it must be surrounded by '', like this: 'filter_tickets_by_file_name="2022_09_15"'
    }

    tickets, entities, labels, _, id_2_label = load_ticket_dataset(**ticket_dataset)
    tickets_no_prompt, _, _, _, _ = load_ticket_dataset(**tickets_no_prompt_dataset)

    output = []

    assert len(tickets) == len(entities) == len(labels) == len(tickets_no_prompt), "wrong lens"

    for ticket_text, entity_list, label, ticket_text_no_prompt in tqdm(
        zip(tickets, entities, labels, tickets_no_prompt), total=len(tickets)
    ):

        text = NER(ticket_text)

        entities_spacy = []
        for word in text.ents:
            entities_spacy.append([word.text, word.label_, word.start_char, word.end_char])

        entity_list = list(map(lambda e: [ticket_text[e[0] : e[1]], e[2], e[0], e[1]], entity_list))

        index_start_ticket_text = ticket_text.find(ticket_text_no_prompt)

        output.append(
            {
                "ticket": ticket_text,
                "entities_spacy": entities_spacy,
                "entities_regex": entity_list,
                "label": id_2_label[label],
                "index_start_ticket_text": index_start_ticket_text,
            }
        )

    with open("util/random_scripts/data/ner_regex.json", "w") as f:
        json.dump(output, f)

    print("SAVED NER TRAIN")


def main():

    NER = spacy.load("en_core_web_md")

    save_survey_spacy_entities(NER)
    # save_train_spacy_entities(NER)


if __name__ == "__main__":
    main()
