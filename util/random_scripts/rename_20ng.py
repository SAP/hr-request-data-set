import os

from tqdm import tqdm


def main():

    for path, _, files in os.walk("ticket_generation/data/20_newsgroups/"):
        for name in tqdm(files):
            if ".txt" not in name:
                file = os.path.join(path, name)
                new_name = file + ".txt"
                os.rename(src=file, dst=new_name)


if __name__ == "__main__":
    main()
