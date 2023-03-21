from __future__ import annotations

import itertools
import re

import pandas as pd
import spacy

from util import entityType

from .text_generator import TicketTextGenerator


class TicketComplaintTextGenerator(TicketTextGenerator):
    def generate_employee(self, employee: pd.Series, **kwargs) -> dict[str, str]:
        employee_new: dict[str, str] = {**employee, "category": self.category, "sub_category": self.sub_category}

        return employee_new

    def get_other_entities(self, ticket: str, employee: dict[str, str]) -> list[entityType]:
        patterns = {"to_who": ["employer", "employers", "supervisor", "supervisors", "employee", "employees"]}

        entities_found_in_ticket_text: list[entityType] = []

        for entity in patterns:
            for pattern in patterns[entity]:
                # Find all match of the feature in the text
                for match in re.finditer(pattern=pattern, string=ticket):
                    _entity_found = (match.start(), match.end(), entity)
                    entities_found_in_ticket_text.append(_entity_found)

        en = spacy.load("en_core_web_sm")
        stopwords = en.Defaults.stop_words

        _complaint = employee["complaint"]
        _complaint_wo_stopwords: list[str] = []

        # Remove stopwords from reason_of_absence
        for token in _complaint.split():
            if token.lower() not in stopwords:
                _complaint_wo_stopwords.append(token)

        # Create every possible ordered combination of complaint and look for
        # it in the text
        #
        # EXAMPLE:
        # "An Urinary Tract Infection" ->
        #       - "Urinary Tract Infection"
        #       - "Urinary Tract"
        #       - "Tract Infection"
        #       - "Urinary"
        #       - "Tract"
        #       - "Infection"
        entity = "complaint"
        for i in reversed(range(1, len(_complaint_wo_stopwords))):
            combinatorics = itertools.combinations(_complaint_wo_stopwords, r=i)
            for combination in combinatorics:
                pattern = " ".join(combination)
                for match in re.finditer(pattern=pattern, string=ticket):
                    _entity_found = (match.start(), match.end(), entity)
                    entities_found_in_ticket_text.append(_entity_found)

        return entities_found_in_ticket_text
