from __future__ import annotations

import datetime
import string
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

LOCALES = ["en-US", "de-DE", "it-IT", "es-ES", "fr-FR"]
LOCALES_CURRENCY = {"en-US": "$", "de-DE": "€", "it-IT": "€", "es-ES": "€", "fr-FR": "€"}
LOCALES_DISTRIBUTION = [0.2, 0.3, 0.15, 0.15, 0.2]


class PersonalInformationFaker:
    """
    This class is responsible for the generation of the personal information of a employee.

    Attributes:
    """

    def __init__(
        self,
        locales: list[str] = LOCALES,
        locales_distribution: list[float] = LOCALES_DISTRIBUTION,
    ):
        assert len(locales_distribution) == len(locales), "locales_distribution and locales must have the same length"
        self._locales = locales
        self._locales_distribution = np.array(locales_distribution) / sum(locales_distribution)
        self.faker = Faker(locales)

    def _generate_single_profile(self) -> dict[str, str]:
        """
        Generates a synthetic profile according to the name and address distributions.

        Returns:
            profile: dictionary containing the candindate personal information.
        """

        _locale: str = np.random.choice(self._locales, p=self._locales_distribution)
        first_name: str = self.faker[_locale].first_name_nonbinary()
        last_name: str = self.faker[_locale].last_name_nonbinary()

        name: str = f"{first_name} {last_name}"
        domain_name: str = self.faker[_locale].domain_name()
        company_name: str
        company_email: str
        company_name, company_email = self._generate_company_info(_locale)

        # Gets $ or € based on location
        currency_symbol: str = LOCALES_CURRENCY[_locale]

        profile: dict[str, str] = {
            "name": name,
            "first_name": first_name,
            "last_name": last_name,
            "nationality": self.faker[_locale].current_country_code(),
            "country": self.faker[_locale].current_country_code(),
            "email": self._generate_email(first_name, last_name, domain_name),
            "company": company_name,
            "company_email": company_email,
            "ticket_date": self._generate_ticket_date(),
            "currency_symbol": currency_symbol,
        }
        return profile

    def _generate_ticket_date(
        self, date_start: Optional[datetime.date] = None, date_end: Optional[datetime.date] = None
    ) -> str:
        """
        Generate random date of creation of a ticket from date_start to date_end.
        Default values are date_end = Now and date_start = 10 years ago

        Args:
            date_start (datetime.date, optional): _description_. Defaults to None.
            date_end (datetime.date, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            str: _description_
        """
        _years: int = 10
        _days_per_year: float = 365.24

        if date_start is None:
            date_start = datetime.datetime.now() - datetime.timedelta(days=_years * _days_per_year)
        if date_end is None:
            date_end = datetime.datetime.now()
        if date_end < date_start:
            raise ValueError("Start date must be smaller than end date")

        _ticket_date: datetime.date = self.faker.date_between_dates(date_start=date_start, date_end=date_end)

        _ticket_date_string: str = _ticket_date.strftime("%d/%m/%Y")

        return _ticket_date_string

    def _generate_company_info(self, locale: str) -> tuple[str, str]:
        """
        Generate company information based on provided locale.

        Args:
            locale: location used for the generation.

        Returns:
            generated company name with corresponding country.
            generated company email: company_name@hr.com ( company_name created removing spaces )
        """

        # Create company
        _company: str = self.faker[locale].company()

        # Create name of the company: company + country
        _company_name: str = f"{_company} {self.faker[locale].current_country()}"

        # Remove punctuation for email
        _company_email: str = _company.translate(str.maketrans("", "", string.punctuation))

        # Remove blank spaces for email
        _company_email = _company_email.replace(" ", "")

        # Add hr email address to email
        _company_email = f"hr@{_company_email}.com"

        return _company_name, _company_email

    def _generate_email(self, first_name: str, last_name: str, domain_name: str) -> str:
        """
        Generates a fake email based on the name of the employee and the given domain name.

        Args:
            first_name: first name of the employee.
            last_name: last name of the employee.
            domain_name: email domain name.

        Returns:
            email: fake email.
        """
        coin: float = np.random.uniform(0, 1)
        email: str
        if coin < 0.5:
            email = f"{first_name}.{last_name}@{domain_name}"
        else:
            email = f"{first_name[0]}{last_name}@{domain_name}"
        return email.lower()

    def generate_profiles(self, size: int = 1) -> pd.DataFrame:
        """
        Generates synthetic profiles.

        Args:
            size: number of profiles to create.

        Returns:
            df_profiles: dataframe containing the personal information of synthetic employees.
        """
        profiles: list[dict[str, str]] = [self._generate_single_profile() for _ in tqdm(range(size))]
        df_profiles: pd.DataFrame = pd.DataFrame(profiles)
        return df_profiles
