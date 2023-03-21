from __future__ import annotations

import glob
import json
import logging
import math
import os
import random
import re
from collections import defaultdict
from typing import Any, TypedDict

from tqdm import tqdm

# from openpyxl import load_workbook


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Entity is ( START_CHAR, END_CHAR, ENTITY_NAME)
entityType = tuple[int, int, str]


class Ticket(TypedDict):
    ticket: str
    category: str
    sub_category: str


def _sample_tickets(tickets: list[dict[str, Any]], sample_size: float) -> list[dict[str, Any]]:
    """
    Sampling of (len(tickets) * sample_size) of tickets and entities

    Args:
        tickets (list[dict[str, Any]]): list of objects of the type {"tickets: "Dear ...", "entities": [[...], [...]]}
        sample_size (float): percentage of sample ( Ex. sample_size=0.6 -> sample 60% of tickets )

    Returns:
        list[dict[str, Any]]: sampled dataset
    """

    if sample_size < 1:
        tickets = random.sample(tickets, math.ceil(len(tickets) * sample_size))

    return tickets


def load_ticket_dataset(
    data_path: str,
    template_path: str,
    template_file_name: str,
    remove_first_part: bool,
    remove_template_sections: bool,
    filter_tickets_by_file_name: str = "",
    sample_size: float = 1.0,
) -> tuple[list[str], list[list[entityType]], list[int], dict[str, int], dict[int, str]]:
    """
    Method used to load tickets from the output of ticket generation.

    Args:
        data_path (str): Path where tickets are saved. In the path there must be the categories'folders present in the object "categories"
        template_path (str): Path where templates are saved
        template_file_name (str): File name of the JSON file where all the templates are saved
        remove_first_part (bool): True if you want to remove from tickets the first part of the ticket, which is not text ( Additional information )
        remove_template_sections (bool): If you want to remove the pre-filled strings belonging to the templates from the final tickets
        filter_tickets_by_file_name (str): Substring of the tickets filter ( Ex. to get tickets generated in the 08/09/2022 -> "08_09_22")
        sample_size (float): Persentage of the dataset to sample ( Must be between 0 and 1 )

    Returns:
        tickets (list[str]): list of read tickets
        entities (list[list[entityType]]): list of all entities found for each ticket
        labels (list[str]): list of tickets labels ( the category they belong to )
        label_2_id (dict[str, int]): dictionary of correspondance of labels to an id integer
        id_2_label (dict[int, str]): dictionary of correspondance of id integers to labels
    """

    assert 0 < sample_size <= 1, "Sample size must be: 0 < sample_size <= 1"

    tickets: list[str] = []
    labels: list[str] = []
    entities: list[list[entityType]] = []

    template_file_path: str = f"{template_path}/{template_file_name}"
    # Load all templates for the categories
    with open(template_file_path) as f:
        templates = json.load(f)

    # Template dictionary with different keys for each category
    templates_tickets: dict = templates["tickets"]

    def get_categories_implemented(templates_tickets: dict) -> list[tuple[str, str]]:
        """
        In file templates.json each record has an 'implemented' feature. The reason is that at the start I wrote
        more prompts than the ones used in the final version. To avoid cancelling all the prompts written
        previously I used this 'implemented' variable. If 'implemented' is set to False than the category is skipped
        """

        list_categories_sub_categories_implemented: list[tuple[str, str]] = []

        for category in templates_tickets:
            for sub_category in templates_tickets[category]:
                if templates_tickets[category][sub_category]["implemented"]:
                    list_categories_sub_categories_implemented.append((category, sub_category))

        return list_categories_sub_categories_implemented

    categories_implemented = get_categories_implemented(templates_tickets)

    for category_name, sub_category_name in categories_implemented:

        _data_folder = f"{data_path}/{templates_tickets[category_name][sub_category_name]['output_folder']}"

        # Get all json files in each folder ( each file has an array of tickets)
        # 'filter_tickets_by_file_name' is used to filter the filenames
        # I mainly uses it to filter from tickets generated in different dates
        file_names: list[str] = [
            f for f in glob.glob(f"{_data_folder}/*{filter_tickets_by_file_name}*.json") if os.path.isfile(f)
        ]

        templates_of_current_category: list[str] = templates_tickets[category_name][sub_category_name]["templates"]

        # Remove from templates the variables strings (ex. "My company is {company_name}" -> "My company is " )
        # and the <generate> keywords
        if remove_template_sections:
            strings_to_remove: list[str] = []
            regex = "\$\{.*\}|\<generate\>"
            for _template in templates_of_current_category:
                _str_to_remove = re.split(
                    regex,
                    _template,
                )

                _str_to_remove: list[str] = [s.strip() for s in _str_to_remove if s != ""]
                strings_to_remove.extend(_str_to_remove)

        for file_name in tqdm(file_names):
            with open(f"{file_name}", encoding="utf8", errors="ignore") as f:
                tickets_objects: list[dict[str, Any]] = json.load(f)

            tickets_objects = _sample_tickets(tickets=tickets_objects, sample_size=sample_size)

            # Load all tickets' texts of the current category
            tickets_loaded: list[str] = map(lambda t: t["ticket"], tickets_objects)

            # Load all entities for all tickets of the current category
            all_entities: list[list[list]] = map(lambda t: t["entities"], tickets_objects)

            #  In JSON there aren't tuples, so I save them as list and convert them here into tuples
            all_entities_tuples: list[list[entityType]] = [
                list(map(lambda l: (l[0], l[1], l[2]), entity_list)) for entity_list in all_entities
            ]

            label: str = f"{category_name}_{sub_category_name}"

            for ticket in tickets_loaded:
                # TODO: change it to smth. more general
                # Right now it looks for the line that starts with "Subject" and removes all the lines
                # before ( included the ones that starts with Subject)
                # This assumes that the last line which is not 'text' ( so the last line of the prompt )
                # is the Subject line.
                # In future the output of the generation model should be changed to divide the two parts,
                # in order to use less 'heuristics' methods to split the two parts
                if remove_first_part:
                    index_line_subject = -1
                    for i, line in enumerate(ticket.split("\n")):
                        if line.strip().startswith("Subject"):
                            index_line_subject = i
                            break

                    index_written_text = index_line_subject + 1
                    ticket = "\n".join(ticket.split("\n")[index_written_text:])

                # Remove from the ticket all the strings coming from the template
                if remove_template_sections:
                    for _str_to_remove in strings_to_remove:
                        ticket = ticket.replace(_str_to_remove, "")

                tickets.append(ticket)
                labels.append(label)

            # Append all entities for each ticket of the current category
            entities.extend(all_entities_tuples)

    # label_2_id: dictionary { category_name: category_id }
    # id_2_label: dictionary { category_id: category_name }
    label_2_id: dict[str, int] = {label: idx for idx, label in enumerate(list(set(labels)))}
    id_2_label: dict[int, str] = {idx: label for label, idx in label_2_id.items()}

    # labels_ids: list of all category ids
    labels_ids: list[int] = list(map(lambda label: label_2_id[label], labels))

    logger.info(f"LOADED {len(tickets)} TICKETS IN TOTAL")

    return tickets, entities, labels_ids, label_2_id, id_2_label


def load_survey_tickets(data_path: str, label_2_id: dict[str, int]) -> tuple[list[str], list[int]]:
    """
    Load tickets gathered from survey

    Args:
        data_path (str): path where json file of survey tickets is saved
        label_2_id (dict[str, int]): dictionary { category_name: category_id }

    Returns:
        tickets_survey_texts (list[str]): list of all the texts of the surveys' tickets
        labels (list[int]): list of the labels of the surveys' tickets
    """

    with open(f"{data_path}/data.json") as f:
        tickets_survey = json.load(f)

    tickets_survey_texts: list[str] = list(map(lambda t: t["ticket"], tickets_survey))
    labels: list[int] = list(map(lambda t: label_2_id[t["label"]], tickets_survey))

    return tickets_survey_texts, labels


def load_survey_tickets_texts(data_path: str) -> list[str]:
    """
    Load tickets(texts only) gathered from survey

    Args:
        data_path (str): path where json file of survey tickets is saved

    Returns:
        list[str]: list of all the texts of the surveys' tickets
    """

    with open(f"{data_path}/data.json") as f:
        tickets_survey = json.load(f)

    tickets_survey_texts: list[str] = list(map(lambda t: t["ticket"], tickets_survey))

    return tickets_survey_texts


def load_survey_tickets_entities(data_path: str) -> tuple[list[str], list[list[entityType]]]:
    """
    Load tickets(texts and entities) gathered from survey

    Args:
        data_path (str): path where json file of survey tickets is saved

    Returns:
        tickets_survey_texts: list of all the texts of the surveys' tickets
        tickets_survey_entities: list of all the entities of the surveys' tickets
    """

    with open(f"{data_path}/data.json") as f:
        tickets_survey = json.load(f)

    tickets_survey_texts: list[str] = list(map(lambda t: t["ticket"], tickets_survey))
    tickets_survey_entities: list[list[entityType]] = list(map(lambda t: t["entities"], tickets_survey))

    return tickets_survey_texts, tickets_survey_entities


def load_survey_tickets_by_category(data_path: str) -> dict[str, list[str]]:
    """
    Load tickets grouped by category

    Args:
        data_path (str): path where json file of survey tickets is saved

    Returns:
        dict[category_name(str), list_of_tickets_texts(list[str])]
    """

    with open(f"{data_path}/data.json") as f:
        tickets_survey = json.load(f)

    ticket_divided_by_category: dict[str, list[Ticket]] = defaultdict(list)

    for ticket in tickets_survey:
        ticket_divided_by_category[ticket["label"]].append(ticket["ticket"])

    return ticket_divided_by_category


def check_ticket_generation_config(conf: dict) -> bool:
    """
    Checks if config for ticket generation respects forced constraints
    This method does not check all constraints, only a few

    Args:
        conf (dict): dictionary of Hydra configs

    Raises:
        ValueError:
        ValueError:

    Returns:
        bool: True if the configs are correct
    """

    num_beams: int = conf.gpt.num_beams
    do_sample: bool = conf.gpt.do_sample
    force_words: str = conf.gpt.force_words

    # Checks taken from https://github.com/huggingface/transformers/blob/main/src/transformers/generation_utils.py

    is_constraint_gen_mode = bool(force_words.strip())

    if is_constraint_gen_mode:
        if num_beams <= 1:
            raise ValueError("`num_beams` needs to be greater than 1 if force_words is not empty.")

        if do_sample:
            raise ValueError("`do_sample` needs to be false for if force_words is not empty.")

    return True
