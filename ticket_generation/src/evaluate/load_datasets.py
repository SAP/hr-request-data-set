from __future__ import annotations

import pandas as pd


class LoadDatasets:
    @staticmethod
    def load_amazon_reviews_text(data_path: str) -> list[str]:
        """
        https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download

        Args:
            data_path (str): _description_

        Returns:
            list[str]: _description_
        """
        df_reviews = pd.read_csv(f"{data_path}/Reviews.csv", nrows=2000)

        return df_reviews["Text"].tolist()

    @staticmethod
    def load_reddit_comments_text(data_path: str) -> list[str]:
        """
        https://www.kaggle.com/datasets/ehallmar/reddit-comment-score-prediction

        Args:
            data_path (str): _description_

        Returns:
            list[str]: _description_
        """

        df_positive_comments = pd.read_csv(f"{data_path}/comments_positive.csv", nrows=1000)
        df_negative_comments = pd.read_csv(f"{data_path}/comments_negative.csv", nrows=1000)

        df_comments = pd.concat([df_positive_comments, df_negative_comments], ignore_index=True)

        return df_comments["text"].tolist()

    @staticmethod
    def load_nips_papers_text(data_path: str) -> list[str]:
        """
        https://www.kaggle.com/datasets/benhamner/nips-2015-papers/versions/2

        Args:
            data_path (str): _description_

        Returns:
            list[str]: _description_
        """
        df_papers = pd.read_csv(f"{data_path}/Papers.csv", nrows=50)

        return df_papers["PaperText"].tolist()
