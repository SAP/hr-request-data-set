import os
import shutil


def main():

    file_paths = []
    with open("ticket_generation/data/enron_mail/sick_leave_match.txt") as f:
        for line in f:
            _file_path = line.split(":")[0][2:]
            _file_path = "ticket_generation/data/enron_mail/maildir/" + _file_path + "txt"
            file_paths.append(_file_path)

    # Remove duplicates
    file_paths = list(set(file_paths))

    name_new_folder = "sick_leave"
    path = "ticket_generation/data/enron_mail/" + name_new_folder

    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)

    for _file_path in file_paths:

        dst = f"{path}/{_file_path.split('/')[-1]}"
        shutil.copy(src=_file_path, dst=dst)


if __name__ == "__main__":
    main()
