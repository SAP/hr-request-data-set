from __future__ import annotations

import itertools
import re

import pandas as pd
import spacy

from util import entityType

from .text_generator import TicketTextGenerator


class TicketLifeEventTextGenerator(TicketTextGenerator):
    def generate_employee(self, employee: pd.Series, **kwargs) -> dict[str, str]:

        employee_new: dict[str, str] = {**employee, "category": self.category, "sub_category": self.sub_category}

        return employee_new

    def get_other_entities(self, ticket: str, employee: dict[str, str]) -> list[entityType]:
        """
        From a ticket text(str) this function find all entities(features) of employee in the text that
        are not exact matches, but that have similar meaning ( Ex. maybe a date is saved in an employee as 01/10/1998 but
        in the text it appears as 1st October 1998, so we would like to mark it also as a date )

        Args:
            ticket (str): ticket text string

        Returns:
            list[entityType]: list of found entities(features) of employee inside the text
                                        format: [( Start_character_index, End_Character_index, Entity_Name), ...]
        """

        entities_found_in_ticket_text: list[entityType] = []

        en = spacy.load("en_core_web_sm")
        stopwords = en.Defaults.stop_words

        _description_life_event = employee["description_life_event"]
        _description_life_event_wo_stopwords: list[str] = []

        # Remove stopwords from reason_of_absence
        for token in _description_life_event.split():
            if token.lower() not in stopwords:
                _description_life_event_wo_stopwords.append(token)

        # Create every possible ordered combination of description_life_event and look for
        # it in the text
        #
        # EXAMPLE:
        # "the divorce from my spouse" ->
        #       - "divorce spouse"
        #       - "divorce"
        #       - "spouse"
        entity = "description_life_event"
        for i in reversed(range(1, len(_description_life_event_wo_stopwords))):
            combinatorics = itertools.combinations(_description_life_event_wo_stopwords, r=i)
            for combination in combinatorics:
                pattern = " ".join(combination)
                for match in re.finditer(pattern=pattern, string=ticket):
                    _entity_found = (match.start(), match.end(), entity)
                    entities_found_in_ticket_text.append(_entity_found)

        return entities_found_in_ticket_text
