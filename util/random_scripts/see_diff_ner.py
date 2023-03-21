import json
from pprint import pprint


def main():

    with open("util/random_scripts/data/ner.json") as f:
        ner = json.load(f)

    with open("util/random_scripts/data/ner_transformers.json") as f:
        ner_trf = json.load(f)

    output = []

    for ticket, ticket_trf in zip(ner, ner_trf):

        assert ticket["id"] == ticket_trf["id"]

        entities_trf = set(tuple(row) for row in ticket_trf["entities_spacy"])
        entities = set(tuple(row) for row in ticket["entities_spacy"])

        not_common_elements = entities_trf.union(entities).difference(entities_trf.intersection(entities))

        if not_common_elements:
            output.append(
                {
                    "id": ticket["id"],
                    "entities": list(entities),
                    "entities_trf": list(entities_trf),
                    "not_common_elements": list(not_common_elements),
                }
            )
            print(f"{entities=}")
            print(f"{entities_trf=}")
            pprint(not_common_elements)

            print("-----------------")

    with open("util/random_scripts/data/ner_diff.json", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
