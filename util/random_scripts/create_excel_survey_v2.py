from __future__ import annotations

import random

import xlsxwriter
from tqdm import tqdm

from util import load_ticket_dataset

"""
From: rdiaz@leger.org
To: hr@HubertRodriguesSA.com
First name: Rémy
Last name: Diaz
Company: Hubert Rodrigues SA France
Date: 03/09/2014
Ticket category: Timetable change
Ticket sub-category: Shift change
Reason of change: I have a wedding
Current shift: Thursday 04 September 22:00
Requested shift: Tuesday 09 September 14:00
Subject: Request of shift change from Thursday 04 September 22:00 to Tuesday 09 September 14:00 because I have a wedding

"""


def preprocess_ticket(ticket: str) -> list[tuple[str, str]]:
    starts_to_delete = ["From:", "To:", "Date:", "Company:"]

    ticket_lines = ticket.split("\n")

    ticket_lines_filtered = [
        line for line in ticket_lines if not any([line.startswith(start) for start in starts_to_delete])
    ]

    ticket_lines_separated: list[tuple[str, str]] = []
    for line in ticket_lines_filtered:

        if not line:
            continue

        split_line = line.split(":")

        detail_name = split_line[0].strip()
        detail_text = split_line[1].strip()

        ticket_lines_separated.append((detail_name, detail_text))

    return ticket_lines_separated


def get_col_widths(column_values: list[str], k: int = 2):
    return max([len(line) + k for line in column_values])


def write_worksheet_ticket_page(worksheet, tickets, bold, input_cell, border, row_to_write_input_ticket):
    worksheet.write("A1", "Ticket details", bold)
    worksheet.write("C1", "Detail used ( X if used, empty if not )", bold)

    ticket_sample: tuple[str, str] = random.choice(tickets)
    for t, detail in enumerate(ticket_sample):
        worksheet.write(2 + t, 0, detail[0], border)
        worksheet.write(2 + t, 1, detail[1], border)
        worksheet.write(2 + t, 2, "", border)

    for column in range(2):
        width = get_col_widths(column_values=[row[column] for row in ticket_sample])
        worksheet.set_column(column, column, width)

    worksheet.write(f"B{row_to_write_input_ticket}", "Ticket text", bold)
    worksheet.write(f"B{row_to_write_input_ticket + 1}", "", input_cell)
    worksheet.set_row(row_to_write_input_ticket, 200)


def main():

    pass

    data_path = "util/random_scripts/data"

    ticket_dataset = {
        "data_path": "ticket_generation/output",
        "template_path": "ticket_generation/data/templates",
        "template_file_name": "templates.json",
        "sample_size": 1.0,  # Must be between 0 and 1, percentage of dataset used
        "remove_first_part": False,
        "remove_template_sections": False,
        "filter_tickets_by_file_name": "2022_10_20",
    }

    print("LOADING TICKETS:")

    tickets: list[str]
    tickets, _, _, _, _ = load_ticket_dataset(**ticket_dataset)

    tickets = list(map(preprocess_ticket, tickets))

    print(f"Number of tickets: {len(tickets)}")

    print("CREATING EXCEL FILE:")

    workbook = xlsxwriter.Workbook(f"{data_path}/ticket_general.xlsx")

    worksheet = workbook.add_worksheet(name=f"data_sheet")

    # tickets = tickets[:20]

    for i, ticket in tqdm(enumerate(tickets)):
        for j, (detail_name, detail_text) in enumerate(ticket):
            worksheet.write(i * 10 + j, 0, detail_name)
            worksheet.write(i * 10 + j, 1, detail_text)

    print(f"Last line number: {i * 10 + j}")

    workbook.close()


if __name__ == "__main__":
    main()
