import pandas as pd

from .employee_generator import EmployeeGenerator
from .personal_information_faker import PersonalInformationFaker


class EmployeeRefundTravelGenerator(EmployeeGenerator):
    """ """

    def __init__(self, personal_info_faker: PersonalInformationFaker = PersonalInformationFaker()):
        self.personal_info_faker = personal_info_faker

    def generate_employees(self, data: pd.DataFrame, size: int = 1) -> pd.DataFrame:
        """
        Generates synthetic employees

        Args:
            size: number of employees to create.

        Returns:
            df: dataframe containing synthetic employees.

        Raises:
            ValueError:
        """
        try:
            df_pi: pd.DataFrame = self.personal_info_faker.generate_profiles(size=size)

            countries: pd.Series = df_pi["country"]
            df_refund_travel: pd.DataFrame = self.generate_refund_travel_df(data=data, size=size, countries=countries)
            df: pd.DataFrame = pd.concat([df_pi, df_refund_travel], axis=1)
            return df
        except ValueError as error:
            raise ValueError("Error") from error

    def generate_refund_travel_df(self, data: pd.DataFrame, size: int, countries: pd.Series) -> pd.DataFrame:
        """
        Args:
            data (pd.DataFrame):
            size (int): number of data we want to create

        Returns:
            pd.DataFrame:
        """

        countries_unique: list[str] = list(countries.unique())

        # Filter flight leaving only from the countries of the employees
        data = data[data["ISO_code_source"].isin(countries_unique)]

        # Get dictionary with keys equal to countries and values equal to list of flights leaving from that country
        locations_by_country: dict[str, pd.DataFrame] = {}
        for _country in countries_unique:
            locations_by_country[_country] = data[data["ISO_code_source"] == _country][
                ["airport_from", "from", "airport_to", "to"]
            ]

        def _get_random_location_by_country(country: str):
            return locations_by_country[country].sample()

        # For each employee get a random flight leaving from his country
        flights_of_employees: pd.DataFrame = pd.concat(
            [_get_random_location_by_country(country) for country in countries]
        )

        new_data: pd.DataFrame = flights_of_employees

        # Reset index to concat later
        new_data = new_data.reset_index()

        return new_data
