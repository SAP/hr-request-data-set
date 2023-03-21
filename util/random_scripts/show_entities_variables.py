import json
import re
from collections import defaultdict
from pprint import pprint

from util import load_ticket_dataset


def print_entities():
    ticket_dataset = {
        "data_path": "ticket_generation/output",
        "template_path": "ticket_generation/data/templates",
        "template_file_name": "templates.json",
        "sample_size": 1,
        "remove_first_part": False,
        "remove_template_sections": False,
        "filter_tickets_by_file_name": "2022_11_16",  # If passed by command line and if it contains underscores it must be surrounded by '', like this: 'filter_tickets_by_file_name="2022_09_15"'
    }

    print(ticket_dataset)

    _, entities, labels, _, id_2_label = load_ticket_dataset(**ticket_dataset)

    categories = defaultdict(list)

    for entity_list, label in zip(entities, labels):
        entity_list = list(map(lambda e: e[2], entity_list))
        categories[id_2_label[label]].extend(entity_list)

    for category in categories:
        categories[category] = set(categories[category])

    pprint(categories)


def print_variables():
    with open("ticket_generation/data/templates/templates.json") as f:
        templates = json.load(f)

    templates = templates["tickets"]

    variables_category = defaultdict(list)

    for category in templates:
        for sub_category in templates[category]:
            if templates[category][sub_category]["implemented"] == True:

                add_info = templates[category][sub_category]["additional_info"]
                templates_cat = templates[category][sub_category]["templates"]
                subject = templates[category][sub_category]["subject"]

                union = [*add_info, *templates_cat, subject]

                for string in union:
                    result = re.findall("\{([^}]+)\}", string)
                    variables_category[f"{category}_{sub_category}"].extend(result)

    for category in variables_category:
        variables_category[category] = set(variables_category[category])

    pprint(variables_category)


def main():
    print_entities()

    print_variables()


if __name__ == "__main__":
    main()
