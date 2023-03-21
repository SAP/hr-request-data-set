import random

import numpy as np
import pandas as pd

from .employee_generator import EmployeeGenerator
from .personal_information_faker import PersonalInformationFaker


class EmployeeInfoAccommodationGenerator(EmployeeGenerator):
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
            df_gender_pay_gap: pd.DataFrame = self.generate_info_accommodation_df(
                data=data, size=size, countries=countries
            )
            df: pd.DataFrame = pd.concat([df_pi, df_gender_pay_gap], axis=1)
            return df
        except ValueError as error:
            raise ValueError("Eror") from error

    def generate_info_accommodation_df(self, data: pd.DataFrame, size: int, countries: pd.Series) -> pd.DataFrame:
        """


        Args:
            data (pd.DataFrame): dataset with location(cities over 100k people) and their respective
                                 countries' codes and names
            size (int): number of data we want to create
            countries (pd.Series): countries of the people generated with Faker

        Returns:
            pd.DataFrame: returns a dataset
        """

        countries_unique: np.ndarray = countries.unique()

        data = data[data["country_code"].isin(countries_unique)]

        # Get dictionary with keys equal to countries and values equal to list of locations
        locations_by_country: dict[str, list[str]] = dict(data.groupby("country_code")["location"].apply(list))

        def _get_random_location_by_country(country: str):
            return random.choice(locations_by_country[country])

        locations_of_employees: pd.Series = countries.map(_get_random_location_by_country)

        # Possible duration of accommodations hard-coded here, to move later
        # durations in months ( 1 month, 6 months, 12 months )
        durations_accommodations = np.array([1, 6, 12])

        duration_accommodations_employees: np.ndarray = np.random.choice(
            durations_accommodations, size=size, replace=True
        )

        def get_duration_text(duration: int):
            return f"{duration} months"

        duration_accommodations_employees_text: list[str] = list(
            map(get_duration_text, duration_accommodations_employees)
        )

        data = pd.DataFrame(
            data={"location": locations_of_employees, "duration": duration_accommodations_employees_text}
        )

        # Reset index to concat later
        data = data.reset_index()

        return data
