import numpy as np
import pandas as pd


def main():
    data = pd.read_csv("ticket_generation/data/geonames-all-cities-with-a-population-1000.csv", sep=";")

    data["Population"] = data["Population"].astype(np.int32)

    cities_population_over_100000 = data[data["Population"] > 100000]

    print(len(data))
    print(len(cities_population_over_100000))

    print(cities_population_over_100000)

    cities_population_over_100000.to_csv(
        "ticket_generation/data/geonames-all-cities-with-a-population-over-100000.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
