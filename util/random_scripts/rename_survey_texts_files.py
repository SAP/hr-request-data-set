import glob
import os


def main():

    data_path = "ticket_generation/data/survey_texts_entities_v3"

    files_survey: list[str] = glob.glob(f"{data_path}/ticket*.txt")

    for old_name in files_survey:

        ticket_name = old_name.split("\\")[1]

        number_txt = ticket_name[ticket_name.index("ticket") + len("ticket") : ticket_name.index(".txt")]

        number = int(number_txt)

        old_name_first_part = old_name.split("\\")[0]

        new_name = f"{old_name_first_part}/ticket_{number:03d}.txt"

        os.rename(old_name, new_name)


if __name__ == "__main__":
    main()
