import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling

from .employee_generator import EmployeeGenerator
from .personal_information_faker import PersonalInformationFaker


class EmployeeBayesianNetwork(BayesianNetwork):
    """
    This class defines the structure of the Bayesian network describing
    the key attributes of an employee and their interdependencies.
    """

    def __init__(self):
        super().__init__()
        self.add_nodes_from(
            [
                "Reason_for_absence",
                "Month_of_absence",
                "Absenteeism_time_in_days",
            ]
        )
        self.add_edges_from(
            [
                ("Reason_for_absence", "Month_of_absence"),
                ("Reason_for_absence", "Absenteeism_time_in_days"),
            ]
        )


class EmployeeAbsenceGenerator(EmployeeGenerator):
    """
    This class is responsible for (privately) learning the conditional probability distributions
    of the input Bayesian network and for generating synthetic employees.

    Attributes:
        bayesian_network: instance of BayesianNetwork (or child class).
        epsilon: privacy budget for differential privacy.
        pseudo_counts: pseudo counts used when learning the conditional probability distributions
                       from data. Larger values decrease the sensitivity of the conditional
                       probabilities and hence reduce the scale of the noise needed for achieving
                       differential privacy.
        personal_info_faker: instance of PersonalInformationFaker.
    """

    def __init__(
        self,
        bayesian_network: BayesianNetwork = EmployeeBayesianNetwork(),
        epsilon: float = 1.0,
        pseudo_counts: float = 1.0,
        personal_info_faker: PersonalInformationFaker = PersonalInformationFaker(),
    ):
        self.bayesian_network = bayesian_network
        self.epsilon = epsilon
        self.pseudo_counts = pseudo_counts
        self.personal_info_faker = personal_info_faker

    def plot(self, node_size: float = 2000.0, font_size: float = 8.0) -> None:
        """
        Plots the input Bayesian network for revision purposes.

        Args:
            node_size: size of the graph nodes.
            font_size: size of the nodes font.
        """
        nx.draw_networkx(
            self.bayesian_network,
            node_size=node_size,
            font_size=font_size,
            pos=nx.circular_layout(self.bayesian_network),
        )

    def learn_cpds(self, data: pd.DataFrame) -> None:
        """
        Learns conditional probability distributions (CPDs) from data.

        Args:
            data: dataframe used for training.

        Raises:
            ValueError: if the dataframe does not contain all the attributes constituting
                        the Bayesian network nodes.
        """
        for attribute in self.bayesian_network.nodes:
            if attribute not in data.columns:
                raise ValueError(f"Input dataframe does not contain field '{attribute}'")

        # Make sure NaNs are converted to empty strings
        df = data.fillna("NA")

        self.bayesian_network.fit(
            df,
            estimator=BayesianEstimator,
            prior_type="dirichlet",
            pseudo_counts=self.pseudo_counts,
            complete_samples_only=False,
        )

    def learn_and_perturb_cpds(self, data: pd.DataFrame) -> None:
        """
        Learns (noisy) conditional probability distributions (CPDs) from data.

        Args:
            data: dataframe used for training.
        """
        self.learn_cpds(data)

        n_attributes: int = len(self.bayesian_network.nodes)
        scale: float = 2 * n_attributes / (self.pseudo_counts * self.epsilon)

        for idx in range(n_attributes):
            noise = np.random.laplace(loc=0, scale=scale, size=self.bayesian_network.cpds[idx].values.shape)
            self.bayesian_network.cpds[idx].values += noise
            self.bayesian_network.cpds[idx].values[self.bayesian_network.cpds[idx].values <= 0] = 1e-6
            self.bayesian_network.cpds[idx].normalize()

    def generate_employees(self, size: int = 1) -> pd.DataFrame:
        """
        Generates synthetic employees combining some fake personal information with the samples
        drawn from the Bayesian network.

        Args:
            size: number of employees to create.

        Returns:
            df: dataframe containing synthetic employees.

        Raises:
            ValueError: if CPDs are not initialized.
        """
        try:
            inference: BayesianModelSampling = BayesianModelSampling(self.bayesian_network)
            df_bn: pd.DataFrame = inference.forward_sample(size=size)
            df_pi: pd.DataFrame = self.personal_info_faker.generate_profiles(size=size)
            df: pd.DataFrame = pd.concat([df_pi, df_bn], axis=1)

            return df
        except ValueError as error:
            raise ValueError("CPDs must be learned first") from error

    def save_model(self, filename: str) -> None:
        """
        Writes the Bayesian network to a BIF file.

        Args:
            filename: the path along with the file name where to write the model.
        """
        self.bayesian_network.save(filename, filetype="bif")
