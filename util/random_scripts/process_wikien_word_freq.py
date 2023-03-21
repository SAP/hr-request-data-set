import numpy as np
import pandas as pd


def main():
    data = pd.read_csv("ticket_generation/data/enwiki-word-freq-2022-08-29.txt", sep=" ", header=None)
    data.columns = ["word", "frequency"]

    data = data[data["frequency"] >= 10]

    data["frequency"] = np.log(data["frequency"])

    data.to_csv("ticket_generation/data/enwiki-word-freq-2022-08-29.csv", index=False)


if __name__ == "__main__":
    main()
