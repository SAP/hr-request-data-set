from __future__ import annotations

import itertools
import json
import random
import re
from datetime import datetime, timedelta

import pandas as pd
import spacy

from util import entityType

from .text_generator import TicketTextGenerator


class TicketAbsenceTextGenerator(TicketTextGenerator):
    """
    This class is responsible for the text generation of synthetic tickets in case of
    a request of sick leave for disease.
    """

    def generate_employee(self, employee: pd.Series, **kwargs) -> dict[str, str]:

        reasons: dict[int, list[str]] = kwargs["absence_reasons"]

        _day: str = "day" if int(employee["Absenteeism_time_in_days"]) == 1 else "days"

        # Pick a random reason of absence given the ICD code
        _reason: str = random.choice(reasons[employee["Reason_for_absence"]])

        _number_of_days: str = f"{(int(employee['Absenteeism_time_in_days']))} {_day}"

        # Pick a random date for start of absence ( Add random number
        # of additional days in range [0,10] to ticket date )
        _ticket_date: datetime = datetime.strptime(employee["ticket_date"], "%d/%m/%Y")
        _days_to_add: int = random.randint(1, 10)
        _date_start_absence: datetime = _ticket_date + timedelta(days=_days_to_add)
        _date_start_absence_text: str = _date_start_absence.strftime("%d/%m/%Y")

        employee_new: dict[str, str] = {
            **employee,
            "category": self.category,
            "sub_category": self.sub_category,
            "reason": _reason,
            "number_of_days": _number_of_days,
            "date_start_absence": _date_start_absence_text,
        }

        return employee_new

    def get_kwargs(self) -> dict[str, dict[int, list[str]]]:
        absence_reasons = self._get_ids_absence_reasons()

        kwargs = {"absence_reasons": absence_reasons}

        return kwargs

    def _get_ids_absence_reasons(self) -> dict[int, list[str]]:
        """
        Get the dictionary of absence reasons composed as (absence_id, list[absence_text]).
        This is needed because in the dataset the absences are saved with their ids.
        For each ICD code, there is a list of diseases associated with the code

        Returns:
            dict[int, list[str]]: _description_
        """
        with open(f"{self.data_path}/ICD_diseases.json") as f:
            absence_reasons = json.load(f)

        # Convert ICD codes from string to integers
        absence_reasons: dict[int, list[str]] = {int(k): v for k, v in absence_reasons.items()}

        return absence_reasons

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
        patterns = {"number_of_days": ["\w+(?=\s+day)\sdays*", "\w+(?=\s+day)\sweeks*", "\w+(?=\s+month)\smonths*"]}

        entities_found_in_ticket_text: list[entityType] = []

        for entity in patterns:
            for pattern in patterns[entity]:
                # Find all match of the feature in the text
                for match in re.finditer(pattern=pattern, string=ticket):
                    _entity_found = (match.start(), match.end(), entity)
                    entities_found_in_ticket_text.append(_entity_found)

        en = spacy.load("en_core_web_sm")
        stopwords = en.Defaults.stop_words

        _reason = employee["reason"]
        _reason_wo_stopwords: list[str] = []

        # Remove stopwords from reason_of_absence
        for token in _reason.split():
            if token.lower() not in stopwords:
                _reason_wo_stopwords.append(token)

        # Create every possible ordered combination of reason_of_absence and look for
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
        entity = "reason"
        for i in reversed(range(1, len(_reason_wo_stopwords))):
            combinatorics = itertools.combinations(_reason_wo_stopwords, r=i)
            for combination in combinatorics:
                pattern = " ".join(combination)
                for match in re.finditer(pattern=pattern, string=ticket):
                    _entity_found = (match.start(), match.end(), entity)
                    entities_found_in_ticket_text.append(_entity_found)

        return entities_found_in_ticket_text
