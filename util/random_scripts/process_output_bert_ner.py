import json


def main():

    with open("util/random_scripts/data/output_test_bert_ner.json") as f:
        outputs = json.load(f)

    with open("ticket_generation/data/survey_tickets/data_new.json") as f:
        survey_tickets = json.load(f)

    new_output = []

    assert len(outputs) == len(survey_tickets), "ERROR: DIFFERENT LEN"

    for output, survey_ticket in zip(outputs, survey_tickets):

        entities_true = survey_ticket["entities"]
        new_output.append({"found_by_bert": [], "true": entities_true})

        tokens = output["tokens"]
        labels = output["labels"]

        for token, label in zip(tokens, labels):
            if label != "O" and label != "PAD":
                new_output[-1]["found_by_bert"].append((token, label))

    # print(new_output)

    with open("util/random_scripts/data/new_output_test_bert_ner.json", "w") as f:
        json.dump(new_output, f)


if __name__ == "__main__":
    main()
