import os


def main():

    for path, _, files in os.walk("ticket_generation/data/enron_mail/maildir"):
        for name in files:
            file = os.path.join(path, name)
            if file[-1] == ".":
                new_name = file.replace(".", ".txt")
                os.rename(src=file, dst=new_name)


if __name__ == "__main__":
    main()
