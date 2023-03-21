from __future__ import annotations

import json
import os

import gdown
from transformers import AutoTokenizer

from .tokenized_dataset import TokenizedDataset


class TopicalDataset(TokenizedDataset):
    def __init__(self, tokenizer: AutoTokenizer, data_path: str):
        super().__init__(tokenizer, data_path)

    def _process_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def _load_dataset(self, data_path: str) -> list[str]:

        name_folder_alexa_dataset = "reading_sets"

        alexa_dataset_path = f"{data_path}/{name_folder_alexa_dataset}"

        if not os.path.isdir(data_path):
            os.mkdir(data_path)

        if not os.path.isdir(alexa_dataset_path):
            url: str = "https://drive.google.com/u/0/uc?id=1YeeLRt0xedS774yLCF6kgRY9y9OhRA1q&export=download"
            output = f"{data_path}/tmp.zip"
            gdown.cached_download(url, path=output, postprocess=gdown.extractall)

            os.remove(output)

        data: list[str] = []
        for root, _, files in os.walk(alexa_dataset_path):
            for file in files:
                if not file.endswith(".json"):
                    continue
                with open(root + "/" + file) as fr:
                    knowledges = json.load(fr)
                    for key in knowledges:
                        knowledge = knowledges[key]
                        for agent in knowledge:
                            if agent == "config":
                                continue
                            elif agent == "article":
                                for k in knowledge[agent]:
                                    if k != "url":
                                        article = knowledge[agent][k]
                                        data.append(article)
                            else:
                                agent_knowledge = knowledge[agent]
                                for agent_k in agent_knowledge:
                                    kk = agent_knowledge[agent_k]
                                    if "shortened_wiki_lead_section" in kk:
                                        shortened_wiki_lead_section = kk["shortened_wiki_lead_section"]
                                        data.append(shortened_wiki_lead_section)

                                    elif "summarized_wiki_lead_section" in kk:
                                        summarized_wiki_lead_section = kk["summarized_wiki_lead_section"]
                                        data.append(summarized_wiki_lead_section)

                                    elif "fun_facts" in kk:
                                        fun_facts = " ".join([fun_fact for fun_fact in kk["fun_facts"]])
                                        data.append(fun_facts)
        return data
