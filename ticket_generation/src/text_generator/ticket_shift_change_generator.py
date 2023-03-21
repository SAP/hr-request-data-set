from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd

from util import entityType

from .text_generator import TicketTextGenerator


class TicketShiftChangeTextGenerator(TicketTextGenerator):
    def generate_employee(self, employee: pd.Series, **kwargs) -> dict[str, str]:

        # Pick a random date for shift change ( Add random number
        # of additional days in range [1,10] to ticket date )
        _ticket_date: datetime = datetime.strptime(employee["ticket_date"], "%d/%m/%Y")

        _days_to_add: int = random.randint(1, 10)
        _date_old_shift: datetime = _ticket_date + timedelta(days=_days_to_add)

        _days_to_add = random.randint(1, 10)
        _date_new_shift: datetime = _date_old_shift + timedelta(days=_days_to_add)

        _date_old_shift_text: str = _date_old_shift.strftime("%A %d %B")
        _date_new_shift_text: str = _date_new_shift.strftime("%A %d %B")

        employee_new: dict[str, str] = {
            **employee,
            "category": self.category,
            "sub_category": self.sub_category,
            "old_date": _date_old_shift_text,
            "new_date": _date_new_shift_text,
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
        # TODO: To implement
        return []

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

        variables_processed["old_date"] = "date"
        variables_processed["new_date"] = "date"

        variables_processed["old_work_shift"] = "work_shift"
        variables_processed["new_work_shift"] = "work_shift"

        return variables_processed
