from collections import defaultdict
from statistics import mean

from fast_bleu import SelfBLEU

from util import load_survey_tickets_by_category, load_ticket_dataset


def main():
    ticket_dataset = {
        "data_path": "ticket_generation/output",
        "template_path": "ticket_generation/data/templates",
        "template_file_name": "templates.json",
        "sample_size": 1,
        "remove_first_part": True,
        "remove_template_sections": False,
        "filter_tickets_by_file_name": "2022_11_16",  # If passed by command line and if it contains underscores it must be surrounded by '', like this: 'filter_tickets_by_file_name="2022_09_15"'
    }

    tickets, _, labels, _, id_2_label = load_ticket_dataset(**ticket_dataset)

    tickets_by_category = defaultdict(list)
    for ticket, label in zip(tickets, labels):
        tickets_by_category[id_2_label[label]].append(ticket.split())
    ticket_survey_by_category = load_survey_tickets_by_category(data_path="ticket_generation/data/survey_tickets")

    for category in tickets_by_category:
        weights = {"bigram": (1 / 2.0, 1 / 2.0)}
        self_bleu_generated = SelfBLEU(tickets_by_category[category], weights)
        self_bleu_survey = SelfBLEU(ticket_survey_by_category[category], weights)

        print(category)
        scores_generated = self_bleu_generated.get_score()
        scores_survey = self_bleu_survey.get_score()
        for key in ["bigram"]:
            print(f"GENERATED {key}: {mean(scores_generated[key])}")
            print(f"SURVEY {key}: {mean(scores_survey[key])}")

        print("-" * 10)


if __name__ == "__main__":
    main()
