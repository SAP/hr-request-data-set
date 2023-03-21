from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Tuple, Union

import numpy as np
import spacy
import torch
from gensim.corpora import Dictionary
from gensim.models import LsiModel, TfidfModel
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    LogitsProcessorList,
    Trainer,
    TrainingArguments,
)

from ticket_generation.src.text_generator.bert_topic_datasets.topical_dataset import TopicalDataset
from ticket_generation.src.text_generator.topic_logit_processor import TopicLogitsProcessor

from .fine_tune_datasets.mail_dataset import MailDataset

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LanguageModel:
    """
    This static class contains a collection of methods for building language models.
    """

    @staticmethod
    def get_gpt2_model_and_tokenizer(
        model_id: str,
        do_sample: bool,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        temperature: float,
        length_penalty: int,
        no_repeat_ngram_size: int,
        bad_words: str,
        force_words: str,
        num_beams: int,
        use_gpu: bool,
        device: str,
    ) -> Tuple[Union[GPT2LMHeadModel, GPTJForCausalLM], AutoTokenizer]:
        """
        Retrieves a GPT model and its corresponding tokenizer.

        Args:
            size: size of the GPT model.
            use_gpu: whether the model should run on GPU.
            top_k: the k most likely next words are filtered and the probability mass is
                   redistributed among only these k words.
            top_p: samples from the smallest possible set of words whose cumulative probability
                   exceeds the probability p.
            repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty
            temperature: the value used to module the logits distribution
            no_repeat_ngram_size: If set to int > 0, all ngrams of that size can only occur
                    once.
            bad_words: List of words separated by commas that are not allowed to be generated.
            force_words: List of words separated by commas that must be generated.
            num_beams: !!! NEEDS TO BE GREATER THAN 1 IF `force_words` ARE NOT EMPTY
            do_sample: !!! NEEDS TO BE FALSE IF `force_words` ARE NOT EMPTY

        Returns:
            model: GPT model.
            tokenizer: tokenizer to be used in combination.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="gen_cache")

        # If the model is gpt-j I need to use the GPTJForCausalLM class, for memory reasons
        if "gpt-j" in model_id:
            model = GPTJForCausalLM.from_pretrained(
                model_id,
                revision="float16",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                cache_dir="gen_cache",
            )
        else:
            model = GPT2LMHeadModel.from_pretrained(model_id, cache_dir="gen_cache")

        model.resize_token_embeddings(len(tokenizer))

        # Need prefix space for bad/force words
        # See https://github.com/huggingface/transformers/issues/14206#issuecomment-955190231
        _tokenizer_for_bad_force_words = AutoTokenizer.from_pretrained(
            model_id, cache_dir="gen_cache", add_prefix_space=True, add_special_tokens=False
        )

        # Get bad and force words list of token ids from strings
        bad_words_ids: Optional[list[list[int]]] = (
            _tokenizer_for_bad_force_words(bad_words.split(",")).input_ids if bad_words else None
        )
        force_words_ids: Optional[list[list[int]]] = (
            _tokenizer_for_bad_force_words(force_words.split(",")).input_ids if force_words else None
        )

        setattr(model, "do_sample_", do_sample)
        setattr(model, "top_k_", top_k)
        setattr(model, "top_p_", top_p)
        setattr(model, "repetition_penalty_", repetition_penalty)
        setattr(model, "temperature_", temperature)
        setattr(model, "length_penalty_", length_penalty)
        setattr(model, "no_repeat_ngram_size_", no_repeat_ngram_size)
        setattr(model, "bad_words_ids_", bad_words_ids)
        setattr(model, "force_words_ids_", force_words_ids)
        setattr(model, "num_beams_", num_beams)

        if use_gpu and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
                model = torch.nn.DataParallel(model)

            model.to(device)

            model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
        else:
            model.to("cpu")

        return model, tokenizer

    @staticmethod
    def get_spacy_model(name: str = "en_core_web_sm") -> spacy.Language:
        """
        Retrieves a spacy model for post-processing.

        Args:
            name: identifier of the spacy model.
        Returns:
            model: spacy language model. If the model is not available, it will try to download it
                   first.
        """
        try:
            model = spacy.load(name)
        except OSError:
            spacy.cli.download(name)
            model = spacy.load(name)
        return model

    @staticmethod
    def add_special_tokens(tokenizer: AutoTokenizer, model: Union[GPT2LMHeadModel, GPTJForCausalLM], special_tokens):
        tokenizer.add_special_tokens(special_tokens)

        model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
        model.resize_token_embeddings(len(tokenizer))

    @staticmethod
    def finetune_model(
        model: Union[GPT2LMHeadModel, GPTJForCausalLM],
        dataset: MailDataset,
        training_arguments: dict,
    ):
        """
        This method finetune the GPT model passed as a parameter on the dataset also passed as a parameter

        Args:
            model (Union[GPT2LMHeadModel, GPTJForCausalLM]): GPT model to finetune
            dataset (MailDataset): Dataset to finetune on
            training_arguments (dict): defined in conf/ticket_generation/config.yaml
            device (str)

        """

        training_args = TrainingArguments(
            **training_arguments,
            report_to="none",
        )

        def data_collator(data: list) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.stack([f[0] for f in data]),
                "attention_mask": torch.stack([f[1] for f in data]),
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        trainer.train()

    @staticmethod
    def _get_topic_word_matrix(tokenizer: AutoTokenizer, data_path: str, num_topics: int, file_topic_matrix_path: str):

        dataset_topic = TopicalDataset(tokenizer=tokenizer, data_path=data_path)

        dataset: list[list[str]] = dataset_topic.get_text_tokens()

        docs = [doc for doc in dataset]
        dictionary = Dictionary(docs)

        no_below = 50
        no_above = 0.2

        corpus_file = f"{data_path}/dictionary.p"
        dictionary_file = f"{data_path}/corpus.pkl"

        dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        corpus = [dictionary.doc2bow(doc) for doc in docs]

        pickle.dump(corpus, open(corpus_file, "wb"))
        pickle.dump(dictionary, open(dictionary_file, "wb"))

        tfidf = TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        lsi_model = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)

        model_file = f"{data_path}/lsi_model_file.pkl"
        lsi_model.save(model_file)

        topic_words = lsi_model.get_topics()  # K X V' (num_topics x selected_vocab_size)
        topic_word_matrix: np.ndarray = np.zeros((num_topics, len(tokenizer)))  # K x V (num_topics x vocab_size)

        for i in range(len(dictionary)):
            j = tokenizer.convert_tokens_to_ids(lsi_model.id2word[i])
            topic_word_matrix[:, j] = topic_words[:, i]

        pickle.dump(topic_word_matrix, open(file_topic_matrix_path, "wb"))

        return topic_word_matrix

    @staticmethod
    def get_logits_processor(
        device: str,
        gamma: int,
        logit_threshold: int,
        topic_index: int,
        num_topics: int,
        vocab_size: int,
        create_topic_word: bool,
        tokenizer: AutoTokenizer,
        data_path: str,
    ) -> LogitsProcessorList:

        file_topic_matrix_path = f"{data_path}/topic_matrix.pkl"

        if os.path.exists(file_topic_matrix_path) and not create_topic_word:
            topic_word_matrix: np.ndarray = pickle.load(open(file_topic_matrix_path, "rb"))
        else:
            topic_word_matrix: np.ndarray = LanguageModel._get_topic_word_matrix(
                tokenizer, data_path, num_topics, file_topic_matrix_path
            )

        assert (
            vocab_size == topic_word_matrix.shape[1]
        ), f"Topic matrix input tokenizer has different vocab size({topic_word_matrix.shape[1]}) than current model({vocab_size})"

        topic_word_vector = torch.from_numpy(topic_word_matrix[topic_index, :]).to(device)

        topic_logits_processor = TopicLogitsProcessor(
            topic_word_vector=topic_word_vector, gamma=gamma, logit_threshold=logit_threshold
        )

        logits_processor_list = LogitsProcessorList([topic_logits_processor])

        return logits_processor_list
