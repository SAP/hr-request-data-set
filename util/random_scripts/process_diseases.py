import csv

diseaseas = []
header = ["ICD_Code", "Disease"]

with open("ticket_generation/data/diseases.txt") as file_txt:
    lines = file_txt.readlines()

    for line in lines:
        diseaseas.append([line.split(" ")[0], " ".join(line.split(" ")[1:]).strip()])


with open("ticket_generation/data/diseases.csv", "w", newline="") as file_csv:
    writer = csv.writer(file_csv)

    writer.writerow(header)

    writer.writerows(diseaseas)
