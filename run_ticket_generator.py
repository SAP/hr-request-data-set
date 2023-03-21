from __future__ import annotations

import logging

import hydra
import pandas as pd
import torch
from hydra.utils import instantiate

from ticket_generation.src.df_provider.df_provider import DfProvider
from ticket_generation.src.employee_generator import EmployeeAbsenceGenerator, EmployeeGenerator
from ticket_generation.src.text_generator import LanguageModel, MailDataset
from ticket_generation.src.text_generator.text_generator import TicketTextGenerator
from util.util import check_ticket_generation_config, entityType

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_models(params, use_gpu, device):
    model, tokenizer = LanguageModel.get_gpt2_model_and_tokenizer(
        model_id=params["model"],
        do_sample=params["do_sample"],
        top_k=params["top_k"],
        top_p=params["top_p"],
        repetition_penalty=params["repetition_penalty"],
        temperature=params["temperature"],
        length_penalty=params["length_penalty"],
        no_repeat_ngram_size=params["no_repeat_ngram_size"],
        bad_words=params["bad_words"],
        force_words=params["force_words"],
        num_beams=params["num_beams"],
        use_gpu=use_gpu,
        device=device,
    )
    spacy_model = LanguageModel.get_spacy_model()

    return model, tokenizer, spacy_model


def create_tickets(cfg):
    """
    Method that creates tickets with their entities.
    The tickets are saved in `ticket_generation/output` in json format


    Args:
        cfg (dict): Dictionary with all the programs params set with Hydra
                    The params are set in the folder conf/ticket_generation
    """
    logger.info("START")

    check_ticket_generation_config(cfg)

    use_gpu: bool = cfg.gpu.use_gpu
    device: str = cfg.gpu.device if use_gpu and torch.cuda.is_available() else "cpu"

    logger.info(f"USE_GPU: {use_gpu and torch.cuda.is_available()}")

    employee_generator: EmployeeGenerator = instantiate(cfg.ticket_type.employee_generator)

    logger.info("GETTING DATASET...")

    # Get dataset from dataset_path
    df_provider: DfProvider = instantiate(cfg.ticket_type.df_provider, dataset_path=cfg.data_creation.data_path)
    data: pd.DataFrame = df_provider.get_dataframe()

    logger.info("CREATING EMPLOYEES...")

    # Absence is the only case in which there is a Bayesian learning
    # Create employees dataset ( with info from Faker and info specific to type of ticket)
    if isinstance(employee_generator, EmployeeAbsenceGenerator):
        employee_generator.learn_cpds(data)
        employees_df: pd.DataFrame = employee_generator.generate_employees(size=cfg.data_creation.number_of_data)
    else:
        employees_df: pd.DataFrame = employee_generator.generate_employees(
            size=cfg.data_creation.number_of_data, data=data
        )

    logger.info("GETTING MODELS...")

    gpt_params: dict = cfg.gpt

    model, tokenizer, spacy_model = get_models(params=gpt_params, use_gpu=use_gpu, device=device)

    # Finetune models
    if cfg.fine_tune.execute:
        # Experiment of finetuning GPT model on a dataset
        # Does not achieve better results, so usually not used

        logger.info("FINETUNING...")

        special_tokens: dict = cfg.fine_tune.special_tokens
        training_arguments: dict = cfg.fine_tune.training_arguments

        LanguageModel.add_special_tokens(tokenizer=tokenizer, model=model, special_tokens=special_tokens)

        mail_dataset = MailDataset(
            tokenizer=tokenizer,
            data_path=f"{cfg.data_creation.data_path}/{cfg.fine_tune.folder}",
            special_tokens=special_tokens,
        )

        LanguageModel.finetune_model(
            model=model,
            dataset=mail_dataset,
            training_arguments=training_arguments,
        )

    logger.info("CREATING TICKETS...")

    logits_processor = []
    if cfg.topic_model_generation.execute:
        # Experiment of creating tickets with topical model generatio
        # See https://arxiv.org/abs/2103.06434

        gamma: int = cfg.topic_model_generation.gamma
        logit_threshold: int = cfg.topic_model_generation.logit_threshold
        topic_index: int = cfg.topic_model_generation.topic_index
        num_topics: int = cfg.topic_model_generation.num_topics
        create_topic_word: bool = cfg.topic_model_generation.create_topic_word
        data_path: str = cfg.topic_model_generation.data_path

        special_tokens: dict = cfg.fine_tune.special_tokens

        LanguageModel.add_special_tokens(tokenizer=tokenizer, model=model, special_tokens=special_tokens)

        logits_processor = LanguageModel.get_logits_processor(
            device=device,
            gamma=gamma,
            logit_threshold=logit_threshold,
            topic_index=topic_index,
            num_topics=num_topics,
            vocab_size=model.vocab_size,
            create_topic_word=create_topic_word,
            tokenizer=tokenizer,
            data_path=data_path,
        )

    create_only_first_part: bool = cfg.data_creation.create_only_first_part

    # Create a ticket text for each employee
    ticket_text_generator: TicketTextGenerator = instantiate(
        cfg.ticket_type.text_generator,
        model=model,
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        spacy_model=spacy_model,
        data_path=cfg.data_creation.data_path,
        create_only_first_part=create_only_first_part,
        min_length=cfg.gpt.min_length,
        word_limit=cfg.gpt.word_limit,
        use_gpu=use_gpu,
        device=device,
    )

    tickets_generated: list[str]
    entities: list[list[entityType]]
    tickets_generated, entities = ticket_text_generator.generate_tickets(employees_df)

    def _log_tickets(tickets):
        for ticket, entity_list in zip(tickets, entities):
            logger.info(f"\n\n{ticket}\n\nEntities: {entity_list}\n\n{'-'*50}")

    # Log tickets'texts to console output ( also in folders multirun or outputs )
    _log_tickets(tickets_generated)

    # Save tickets in JSON format in ticket_generation/output
    ticket_text_generator.save_tickets_and_entities_to_file(
        tickets=tickets_generated,
        entities=entities,
        ticket_type=cfg.ticket_type.name,
        output_path=cfg.data_creation.output_path,
    )


@hydra.main(config_path="conf/ticket_generation", config_name="config")
def main(cfg):
    """
    Main method to generate new synthetic tickets

    Args:
        cfg (dict): Dictionary with all the programs params set with Hydra
                    The params are set in the folder conf/ticket_generation
    """

    create_tickets(cfg)


if __name__ == "__main__":
    main()
