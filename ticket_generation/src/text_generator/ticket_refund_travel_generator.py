from __future__ import annotations

import random
import re
from datetime import datetime, timedelta

import pandas as pd

from util import entityType

from .text_generator import TicketTextGenerator


class TicketRefundTravelTextGenerator(TicketTextGenerator):
    def generate_employee(self, employee: pd.Series, **kwargs) -> dict[str, str]:

        # Pick a random date for travel ( Subtract random number
        # of days in range [1,30] to ticket date )
        _ticket_date: datetime = datetime.strptime(employee["ticket_date"], "%d/%m/%Y")
        _days_to_subtract: int = random.randint(1, 30)
        _date_travel: datetime = _ticket_date - timedelta(days=_days_to_subtract)
        _date_travel_text: str = _date_travel.strftime("%d/%m/%Y")

        employee_new: dict[str, str] = {
            **employee,
            "category": self.category,
            "sub_category": self.sub_category,
            "date_travel": _date_travel_text,
        }

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

        _date_travel: datetime = datetime.strptime(employee["date_travel"], "%d/%m/%Y")

        # Taken from here: https://stackoverflow.com/a/5891598
        # Convert date from format 01/10/1998 to 1st October 1998 and to October 1st 1998
        def suffix(d: int) -> str:
            return "th" if 11 <= d <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")

        def custom_strftime(format: str, t: datetime) -> str:
            return t.strftime(format).replace("{S}", str(t.day) + suffix(t.day))

        patterns = [custom_strftime("%B {S} %Y", _date_travel), custom_strftime("%B {S} %Y", _date_travel)]

        entities_found_in_ticket_text: list[entityType] = []

        entity = "date_travel"
        for pattern in patterns:
            # Find all match of the feature in the text
            for match in re.finditer(pattern=pattern, string=ticket):
                _entity_found = (match.start(), match.end(), entity)
                entities_found_in_ticket_text.append(_entity_found)

        return entities_found_in_ticket_text

    def process_variables(self, variables: list[str]) -> dict[str, str]:
        """
        Method to make variables that have different name the same
        Example: "airport_form", "airport_to" --> both become "airport"
        In case there is no variables' names to be changed, the method is not reimplemented in
        the inherited classes.

        Args:
            variables (list[str]): list of variables

        Returns:
            dict[str, str]: dictionary with key equal to old name and value equal to new name
                            most of the variables will haave the same old and new name
        """
        variables_processed = super().process_variables(variables)

        variables_processed["from"] = "location"
        variables_processed["to"] = "location"

        variables_processed["airport_from"] = "airport"
        variables_processed["airport_to"] = "airport"

        return variables_processed
