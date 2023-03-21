from __future__ import annotations

import re

import pandas as pd

from util import entityType

from .text_generator import TicketTextGenerator


class TicketSalaryTextGenerator(TicketTextGenerator):
    def generate_employee(self, employee: pd.Series, **kwargs) -> dict[str, str]:

        employee["prev_salary_text"] = f"{employee['currency_symbol']}{employee['prev_salary']}"
        employee["new_salary_text"] = f"{employee['currency_symbol']}{employee['new_salary']}"

        employee_new: dict[str, str] = {
            **employee,
            "category": self.category,
            "sub_category": self.sub_category,
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

        # Remove currency character from string and convert it into int
        _prev_salary: int = int(employee["prev_salary_text"][1:])

        # Remove currency character from string and convert it into int
        _new_salary: int = int(employee["new_salary_text"][1:])

        patterns = [
            f"{_prev_salary}",
            f"{_prev_salary:,}",
            f"{_new_salary}",
            f"{_new_salary:,}",
            f"{_new_salary:,}".replace(",", " "),
        ]

        entity = "salary"
        for pattern in patterns:
            # Find all match of the feature in the text
            for match in re.finditer(pattern=pattern, string=ticket):
                _entity_found = (match.start(), match.end(), entity)
                entities_found_in_ticket_text.append(_entity_found)

        # Remove '%' character from string and convert it into int
        _increase_in_percentage: int = int(employee["increase_in_percentage"][:-1])

        patterns = [
            f"{_increase_in_percentage + 1}%",
            f"{_increase_in_percentage - 1}%",
            f"{_increase_in_percentage} %",
            f"{_increase_in_percentage + 1} %",
            f"{_increase_in_percentage - 1} %",
            f"{_increase_in_percentage} percent",
            f"{_increase_in_percentage + 1} percent",
            f"{_increase_in_percentage - 1} percent",
        ]

        entity = "increase_in_percentage"
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

        variables_processed["prev_salary_text"] = "salary"
        variables_processed["new_salary_text"] = "salary"

        return variables_processed
