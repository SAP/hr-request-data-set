import json


def main():

    categories_implemented = [
        {
            "category": "Life event",
            "sub_category": "Health issues",
        },
        {
            "category": "Salary",
            "sub_category": "Salary raise",
        },
        {
            "category": "Salary",
            "sub_category": "Gender pay gap",
        },
        {
            "category": "Life event",
            "sub_category": "Personal issues",
        },
        {
            "category": "Complaint",
            "sub_category": "complaint",
        },
        {
            "category": "Refund",
            "sub_category": "Refund travel",
        },
        {
            "category": "Ask information",
            "sub_category": "Accommodation",
        },
    ]

    ticket_categories = [f"{category['category']}_{category['sub_category']}" for category in categories_implemented]

    number_of_tickets = 70
    tickets_for_category = number_of_tickets // len(ticket_categories)

    data = [
        {"id": i, "ticket": "", "label": ticket_categories[i // tickets_for_category]}
        for i in range(number_of_tickets)
    ]

    with open("ticket_handwritten.json", "w") as f:
        json.dump(data, f)
        print("New json file is created")


if __name__ == "__main__":
    main()
