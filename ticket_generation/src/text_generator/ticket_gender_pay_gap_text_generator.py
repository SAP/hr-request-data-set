from __future__ import annotations

import re

import pandas as pd

from util import entityType

from .text_generator import TicketTextGenerator


class TicketGenderPayGapTextGenerator(TicketTextGenerator):
    def generate_employee(self, employee: pd.Series, **kwargs) -> dict[str, str]:

        _wage_gap: str = f"{employee['DiffMeanHourlyPercent']}%"

        employee_new: dict[str, str] = {
            **employee,
            "category": self.category,
            "sub_category": self.sub_category,
            "wage_gap": _wage_gap,
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

        # Remove '%' character from string and convert it into float
        _wage_gap: float = float(employee["wage_gap"][:-1])

        patterns = [
            f"{_wage_gap:.0f}%",
            f"{_wage_gap:.1f}%",
            f"{_wage_gap:.2f}%",
            f"{_wage_gap + 1:.0f}%",
            f"{_wage_gap - 1:.0f}%",
            f"{_wage_gap:.0f} %",
            f"{_wage_gap:.1f} %",
            f"{_wage_gap:.2f} %",
            f"{_wage_gap + 1:.0f} %",
            f"{_wage_gap - 1:.0f} %",
            f"{_wage_gap:.0f} percent",
            f"{_wage_gap + 1:.0f} percent",
            f"{_wage_gap - 1:.0f} percent",
        ]

        entity = "wage_gap"
        for pattern in patterns:
            # Find all match of the feature in the text
            for match in re.finditer(pattern=pattern, string=ticket):
                _entity_found = (match.start(), match.end(), entity)
                entities_found_in_ticket_text.append(_entity_found)

        return entities_found_in_ticket_text
