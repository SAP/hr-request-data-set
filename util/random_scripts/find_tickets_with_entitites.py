import csv
from collections import Counter
from pprint import pprint

from tqdm import tqdm

from util import load_ticket_dataset


def main():
    ticket_dataset = {
        "data_path": "ticket_generation/output",
        "template_path": "ticket_generation/data/templates",
        "template_file_name": "templates.json",
        "sample_size": 1,
        "remove_first_part": False,
        "remove_template_sections": False,
        "filter_tickets_by_file_name": "2022_11_16",  # If passed by command line and if it contains underscores it must be surrounded by '', like this: 'filter_tickets_by_file_name="2022_09_15"'
    }

    ticket_dataset_no_prompt = {
        "data_path": "ticket_generation/output",
        "template_path": "ticket_generation/data/templates",
        "template_file_name": "templates.json",
        "sample_size": 1,
        "remove_first_part": True,
        "remove_template_sections": False,
        "filter_tickets_by_file_name": "2022_11_16",  # If passed by command line and if it contains underscores it must be surrounded by '', like this: 'filter_tickets_by_file_name="2022_09_15"'
    }

    tickets, entities, labels, _, _ = load_ticket_dataset(**ticket_dataset)
    tickets_no_prompt, _, _, _, _ = load_ticket_dataset(**ticket_dataset_no_prompt)

    count_tickets_entities_in_text = Counter()
    count_tickets_with_entities_in_text = 0

    for ticket, entity_list, ticket_no_prompt, label in tqdm(
        zip(tickets, entities, tickets_no_prompt, labels), total=len(tickets)
    ):

        found_entity = False

        index_start_ticket_text = ticket.find(ticket_no_prompt)

        for entity in entity_list:
            if entity[0] > index_start_ticket_text:
                count_tickets_entities_in_text[entity[2]] += 1
                found_entity = True

        if found_entity:
            count_tickets_with_entities_in_text += 1

    pprint(count_tickets_entities_in_text)
    print(f"{count_tickets_with_entities_in_text=}")

    list_entities_count = [[key, value] for key, value in count_tickets_entities_in_text.items()]

    with open("util/random_scripts/data/list_entities_count.csv", "w") as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerows(list_entities_count)


if __name__ == "__main__":
    main()
