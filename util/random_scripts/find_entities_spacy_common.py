from __future__ import annotations

import json
from collections import Counter
from pprint import pprint


def get_common_entities(
    entites_spacy: list,
    entities_regex: list,
    entity_to_include_regex,
    entity_to_include_spacy,
    index_start_ticket_text: int,
):
    """
    Adapted from: https://www.geeksforgeeks.org/find-intersection-of-intervals-given-by-two-lists/

    Args:
        entites_spacy (list): _description_
        entities_regex (list): _description_
    """
    # i and j pointers for arr1
    # and arr2 respectively
    i = j = 0

    n = len(entites_spacy)
    m = len(entities_regex)

    common_entities = []

    # Loop through all intervals unless one
    # of the interval gets exhausted
    while i < n and j < m:

        # Left bound for intersecting segment
        l = max(entites_spacy[i][2], entities_regex[j][2])

        # Right bound for intersecting segment
        r = min(entites_spacy[i][3], entities_regex[j][3])

        # If segment is valid print it
        if (
            l <= r
            and entites_spacy[i][1] in entity_to_include_spacy
            and entities_regex[j][1] in entity_to_include_regex
            and l > index_start_ticket_text
        ):
            common_entities.append([*entites_spacy[i], entities_regex[j][1]])

        # If i-th interval's right bound is
        # smaller increment i else increment j
        if entites_spacy[i][3] < entities_regex[j][3]:
            i += 1
        else:
            j += 1

    return common_entities


def main():
    entities_we_care = {
        "Ask information_Accommodation": {"location": "GPE", "duration": "DATE"},
        "Life event_Health issues": {"date_start_absence": "DATE", "number_of_days": "DATE"},
        "Refund_Refund travel": {"date_travel": "DATE", "location": "GPE"},
        "Salary_Gender pay gap": {"wage_gap": "PERCENT"},
        "Salary_Salary raise": {"increase_in_percentage": "PERCENT", "salary": "MONEY"},
        "Timetable change_Shift change": {"date": "DATE"},
    }

    with open("util/random_scripts/data/ner_regex.json") as f:
        tickets = json.load(f)

    categories_count = Counter()

    for ticket in tickets:
        entities_regex = ticket["entities_regex"]
        entites_spacy = ticket["entities_spacy"]
        label = ticket["label"]
        index_start_ticket_text = ticket["index_start_ticket_text"]

        if label not in entities_we_care.keys():
            continue

        entity_to_include_regex = entities_we_care[label].keys()
        entity_to_include_spacy = entities_we_care[label].values()

        common_entities = get_common_entities(
            entites_spacy, entities_regex, entity_to_include_regex, entity_to_include_spacy, index_start_ticket_text
        )

        if len(common_entities) > 1:
            categories_count[label] += 1

    pprint(categories_count)


if __name__ == "__main__":
    main()
