import pandas as pd


def main():
    routes_df = pd.read_csv("ticket_generation/data/routes.csv")
    routes_df = routes_df[["Source airport", "Destination airport"]]

    airports_df = pd.read_csv("ticket_generation/data/airports.csv")  # IATA code
    airports_df = airports_df[["Name", "City", "Country", "IATA"]]

    countries_df = pd.read_csv("ticket_generation/data/countries.csv")
    countries_df = countries_df[["Country_Name", "ISO_code"]]

    airports_df = pd.merge(airports_df, countries_df, left_on="Country", right_on="Country_Name")

    flight_source_df = pd.merge(
        routes_df,
        airports_df,
        left_on="Source airport",
        right_on="IATA",
    )

    flight_source_df = pd.merge(
        flight_source_df, airports_df, left_on="Destination airport", right_on="IATA", suffixes=("_source", "_dest")
    )

    flight_source_df.to_csv("ticket_generation/data/flights.csv", index=False)


if __name__ == "__main__":
    main()
