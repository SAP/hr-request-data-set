from __future__ import annotations

import re
from statistics import mean, stdev
from typing import TypedDict

import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


class EvaluationsText(TypedDict):
    avg_ttr_unigram: float
    avg_ttr_bigram: float
    avg_ratios_of_noun_phrases: float
    avg_ratios_of_verbs_phrases: float
    avg_word_frequency: float
    average_word_count: float


class EvaluateText:
    """
    Class to calculate different metrics to evaluate text
    """

    def __init__(self, word_frequency_path: str):
        """
        Class that is used to do unsupervised evaluations of a corpora of text
        Main method is compute_all_evaluations

        Args:
            word_frequency_path (str): file path od wikipedia dump word frequency file ( frequencies are
                                       preprocessed with log_e )

        """
        _word_frequency_df = pd.read_csv(word_frequency_path)

        self.word_frequency_dict = {word: freq for word, freq in _word_frequency_df.values}

    def get_word_counts(self, tickets: list[str]) -> list[float]:
        """
        Calculate the average word count of a collection of tickets

        Args:
            tickets (list[str]): list of tickets

        Returns:
            float: word count for each ticket
        """

        average_word_count = [len(ticket.split()) for ticket in tickets]

        return average_word_count

    def get_type_token_ratio_unigrams(self, tickets: list[list[str]]) -> list[float]:
        """
        Calculate the TTR(type token ratio) for the unigrams of the tickets
        TTR = number_of_unique_unigrams / total_number_of_unigrams

        Args:
            tickets (list[list[str]]): list or tickets already tokenized ( each ticket is a list of token )

        Returns:
            list[float]: TTR(type token ratio) for each ticket
        """

        ttr_unigram: list[float] = [len(set(_ticket)) / len(_ticket) for _ticket in tickets if len(_ticket) > 0]

        return ttr_unigram

    def get_type_token_ratio_bigrams(self, tickets: list[list[str]]) -> list[float]:
        """
        Calculate the TTR(type token ratio) for the bigrams of the tickets
        TTR = number_of_unique_bigrams / total_number_of_bigrams

        Args:
            tickets (list[list[str]]): list or tickets already tokenized ( each ticket is a list of token )

        Returns:
            list[float]: TTR(type token ratio) for each ticket

        """

        bigrams_all: list[list[tuple]] = [list(ngrams(_ticket, 2)) for _ticket in tickets]
        ttr_bigram: list[float] = [len(set(bigrams)) / len(bigrams) for bigrams in bigrams_all if len(bigrams) > 0]

        return ttr_bigram

    def get_avg_word_frequencies(self, tickets: list[str]) -> list[float]:
        word_frequencies_avgs: list[float] = []

        for ticket in tickets:

            ticket_splitted: list[str] = ticket.split()
            _word_frequencies: list[int] = []

            for word in ticket_splitted:
                word_lower = word.lower()
                if word_lower in self.word_frequency_dict.keys():
                    # Check if it is present in wikipedia dump of (Word, log_e(Frequency))
                    _word_frequencies.append(self.word_frequency_dict[word_lower])

            if len(_word_frequencies) > 0:
                word_frequencies_avgs.append(mean(_word_frequencies))

        # Necessary to make the avg method called afterwards not to fail
        if len(word_frequencies_avgs) == 0:
            word_frequencies_avgs.append(0)

        return word_frequencies_avgs

    def get_pos_ratios(self, tickets: list[list[tuple[str, str]]], pos: str) -> list[float]:
        """
        Calculate the ratio of a pos(Part Of Speech) for each ticket
        Ex. pos="NOUN"
            pos_ratio = counter_noun_words / counter_all_words

        Args:
            tickets (list[list[tuple[str, str]]]): for each ticket there is a list of all the tokens texts and pos types
                                                   Ex. [ ("I", "PRON"), ("love", "VERB"), ("apples", "NOUN") ]
            pos (str): Part of Speech identifier of spacy
                       ( Complete list: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py#L22 )

        Returns:
            list[float]: ratios of a pos(Part Of Speech) for each ticket
        """

        pos_ratios: list[float] = []

        for _ticket in tickets:

            if len(_ticket) == 0:
                continue

            _counter_noun_phrase = 0
            for _, token_pos in _ticket:
                if token_pos == pos:
                    _counter_noun_phrase += 1

            pos_ratio = _counter_noun_phrase / len(_ticket)
            pos_ratios.append(pos_ratio)

        return pos_ratios

    def get_tokenized_tickets(self, tickets: list[str]) -> list[list[str]]:
        """
        Remove punctuation and split tickets strings in tokens

        Args:
            tickets (list[str]): list of tickets

        Returns:
            list[list[str]]: for each ticket list of tokens of the ticket
        """
        # Remove punctuation
        _tickets_wo_punct = [re.sub(r"[^\w\s]", "", ticket) for ticket in tickets]

        tokenized_ticket = [word_tokenize(_ticket) for _ticket in _tickets_wo_punct]

        return tokenized_ticket

    def get_spacy_tokenized_tickets(self, tickets: list[str]) -> list[list[tuple[str, str]]]:
        """
        For each ticket it splits the ticket in tokens and save the tuple (token, pos) where pos means Part Of Speech

        Args:
            tickets (list[str]): list of tickets

        Returns:
            list[list[tuple[str, str]]]: the list of tuples (word, pos) for each ticket
        """

        nlp = spacy.load("en_core_web_sm")

        _doc_tickets = [nlp(_ticket) for _ticket in tickets]

        _tokenized_spacy_tickets: list[list[tuple[str, str]]] = []
        for doc in _doc_tickets:
            _tokenized_spacy_tickets.append([])
            for token in doc:
                _tokenized_spacy_tickets[-1].append((token.text, token.pos_))

        return _tokenized_spacy_tickets

    def compute_all_evaluations(self, tickets: list[str]) -> EvaluationsText:
        """
        Compute all defined evaluations for a list of tickets

        Returns:
            EvaluationsText: dict of evalations
        """

        tokenized_tickets: list[list[str]] = self.get_tokenized_tickets(tickets=tickets)
        tokenized_tickets_spacy: list[list[tuple[str, str]]] = self.get_spacy_tokenized_tickets(tickets=tickets)

        ttr_unigram: list[float] = self.get_type_token_ratio_unigrams(tickets=tokenized_tickets)
        ttr_bigram: list[float] = self.get_type_token_ratio_bigrams(tickets=tokenized_tickets)

        # Here I pass not tokenized tickets
        avg_word_frequencies: list[float] = self.get_avg_word_frequencies(tickets=tickets)
        word_counts: float = self.get_word_counts(tickets=tickets)

        POS_NOUN_SPACY = "NOUN"
        ratios_of_noun_phrases: list[float] = self.get_pos_ratios(tokenized_tickets_spacy, pos=POS_NOUN_SPACY)
        POS_VERB_SPACY = "VERB"
        ratios_of_verbs_phrases: list[float] = self.get_pos_ratios(tokenized_tickets_spacy, pos=POS_VERB_SPACY)

        avg_ttr_unigram: float = mean(ttr_unigram)
        avg_ttr_bigram: float = mean(ttr_bigram)
        avg_ratios_of_noun_phrases: float = mean(ratios_of_noun_phrases)
        avg_ratios_of_verbs_phrases: float = mean(ratios_of_verbs_phrases)

        avg_word_frequency: float = mean(avg_word_frequencies)
        avg_word_count: float = mean(word_counts)
        stdev_word_frequency: float = stdev(avg_word_frequencies)
        stdev_word_count: float = stdev(word_counts)

        return {
            "avg_ttr_unigram": f"{avg_ttr_unigram:.2f}",
            "avg_ttr_bigram": f"{avg_ttr_bigram:.2f}",
            "avg_ratios_of_noun_phrases": f"{avg_ratios_of_noun_phrases:.2f}",
            "avg_ratios_of_verbs_phrases": f"{avg_ratios_of_verbs_phrases:.2f}",
            "avg_word_frequency": f"{avg_word_frequency:.2f} | stdev: {stdev_word_frequency:.2f}",
            "average_word_count": f"{avg_word_count:.2f} | stdev: {stdev_word_count:.2f}",
        }
