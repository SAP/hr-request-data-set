import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm


def save_models(data, data_path, file_names):

    topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
    topics, probs = topic_model.fit_transform(data)

    freq = topic_model.get_topic_info()
    freq.to_csv(f"{data_path}/freq.csv")

    probs_df = pd.DataFrame(probs)
    probs_df["file_names"] = file_names

    probs_df.to_csv(f"{data_path}/probs.csv")

    print(type(freq))
    print(freq)

    print(topics)


def divide_dataset(data_df):
    religion = {
        12: "12_atheists_god_atheism_atheist",
        26: "16_hell_god_eternal_jesus",
        21: "21_islam_quran_islamic_rushdie",
        70: "70_existence_evolution_exist_god",
        77: "77_ra_satan_god_thou",
        99: "99_pope_church_schism_orthodox",
        146: "146_religion_motivated_religously_religiously",
        192: "191_cancer_burzynskis_a10_tumor",
    }
    health_issue = {
        15: "15_health_tobacco_cesarean_smokeless",
        20: "20_pain_migraine_drug_cancer",
        39: "39_candida_yeast_infection_infections",
        73: "73_insurance_geico_car_accident",
        85: "85_alzheimers_medical_disease_medicine",
        112: "112_cancer_medical_center_centers",
        138: "138_polio_patients_postpolio_muscle",
    }
    lgbt = {
        27: "27_kinsey_sex_men_homosexual",
        31: "31_homosexuality_homosexual_paul_homosexuals",
        107: "107_sexual_enviroleague_gay_homosexuals",
    }
    drugs = {64: "64_drugs_drug_cocaine_marijuana", 211: "211_prozac_zoloft_effects_serotonin"}
    race = {29: "29_blacks_penalty_death_punishment"}

    topics = {"religion": religion, "health_issue": health_issue, "lgbt": lgbt, "drugs": drugs, "race": race}

    data_path = "ticket_generation/data/20_newsgroups"
    probs_df = pd.read_csv(f"{data_path}/probs.csv", index_col=0)
    freq = pd.read_csv(f"{data_path}/freq.csv", index_col=0)

    # delete unknown topic
    freq = freq.iloc[1:, :]

    # reset index
    freq = freq.reset_index(drop=True)

    # argsort by highest prob for each document
    indices_top_topics = probs_df.iloc[:, :-1].values.argsort(axis=1)

    # get top_n prob indeces for each document
    top_n = 3
    indices_top_topics = indices_top_topics[:, -top_n:]

    file_names = probs_df.iloc[:, -1]

    documents = defaultdict(list)

    # For every document top probability, I look if they are in the topics keys list
    # If yes, I save the document for the given topic
    for indice_top_topics, file_name in zip(indices_top_topics, file_names):
        for key in topics:
            if np.any(np.in1d(indice_top_topics, np.array(list(topics[key].keys())))):
                documents[key].append(file_name)

    # save in separated folder the new topics
    for key in documents:
        for file_path in tqdm(documents[key]):
            # src = f'{data_path}/{"/".join(file_path.split("/")[-2:])}.txt'

            file = data_df.loc[data_df["filename"].str.contains(file_path.split("/")[-1])]
            file = file.iloc[0]["text"]

            file_name = file_path.split("/")[-1]
            dst = f"{data_path}/{key}/{file_name}.txt"
            # shutil.copyfile(src, dst)
            with open(dst, "w", encoding="utf8", errors="ignore") as f:
                f.write(file)


if __name__ == "__main__":

    data_path = "ticket_generation/data/enron_mail/maildir"

    # Pick only emails in sent folders
    file_names = [f for f in tqdm(glob.glob(f"{data_path}/*/*sent*/*"))]
    file_names = [f for f in tqdm(file_names) if os.path.isfile(f)]
    data = []

    for file_name in tqdm(file_names):
        with open(f"{file_name}", encoding="utf8", errors="ignore") as f:
            data.append(f.read())

    save_models(data, data_path, file_names)

    data_df = pd.DataFrame(
        {
            "text": data,
            "filename": file_names,
        }
    )

    # divide_dataset(data_df)
