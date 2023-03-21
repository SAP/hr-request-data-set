from __future__ import annotations

import json

import numpy as np
import torch
from keras_preprocessing.sequence import pad_sequences
from seqeval.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from transformers import AdamW, BertForTokenClassification, BertTokenizer, get_linear_schedule_with_warmup

from util import load_survey_tickets_entities, load_ticket_dataset


def tokenize_and_preserve_labels(sentence, entities, tag2idx, tokenizer):

    ordered_entities = sorted(entities, key=lambda e: e[0])

    splitted_sentence = sentence.split(" ")
    count_letters = 0
    count_entities = 0

    final_tokens = []
    labels = []
    first = True

    for word in splitted_sentence:
        for token in tokenizer.tokenize(word):
            if token == "*":
                count_letters += 1
                continue

            if count_letters > ordered_entities[count_entities][1] and count_entities < len(ordered_entities) - 1:
                count_entities += 1

            if count_letters in range(ordered_entities[count_entities][0], ordered_entities[count_entities][1]):
                if first:
                    entity = tag2idx[f"B-{ordered_entities[count_entities][2]}"]
                    first = False
                else:
                    entity = tag2idx[f"I-{ordered_entities[count_entities][2]}"]
                labels.append(entity)
            else:
                labels.append(tag2idx["O"])
                first = True

            final_tokens.append(token)
            for letter in token:
                if letter != "#":
                    count_letters += 1

        count_letters += 1  # Space character

    return final_tokens, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.device_count()

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False, add_prefix_space=True)

    all_entities = {
        "Ask information_Accommodation": {"location", "duration"},
        "Complaint_complaint": {"complaint", "to_who", "reason"},
        "Life event_Health issues": {"date_start_absence", "number_of_days", "reason"},
        "Life event_Personal issues": {"description_life_event"},
        "Refund_Refund travel": {"airport", "date_travel", "location"},
        "Salary_Gender pay gap": {"wage_gap"},
        "Salary_Salary raise": {"increase_in_percentage", "salary", "work_title"},
        "Timetable change_Shift change": {"date", "reason_of_change", "work_shift"},
    }

    list_entities = []

    for key in all_entities:
        for entity in all_entities[key]:
            list_entities.append(entity)

    BIO_entities = []
    for entity in list_entities:
        BIO_entities.append(f"B-{entity}")
        BIO_entities.append(f"I-{entity}")

    BIO_entities.append("PAD")
    BIO_entities.append("O")  # Other

    BIO_entities = list(set(BIO_entities))

    tag2idx = {t: i for i, t in enumerate(BIO_entities)}

    def preprocess_sentence(sentence):
        return sentence.replace("\n", "*")

    ticket_dataset = {
        "data_path": "ticket_generation/output",
        "template_path": "ticket_generation/data/templates",
        "template_file_name": "templates.json",
        "sample_size": 1,
        "remove_first_part": False,
        "remove_template_sections": False,
        "filter_tickets_by_file_name": "2022_11_16",  # If passed by command line and if it contains underscores it must be surrounded by '', like this: 'filter_tickets_by_file_name="2022_09_15"'
    }

    tickets, entities, _, _, _ = load_ticket_dataset(**ticket_dataset)

    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(preprocess_sentence(sent), labs, tag2idx, tokenizer)
        for sent, labs in zip(tickets, entities)
    ]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    MAX_LEN = 512
    batch_size = 16

    input_ids = pad_sequences(
        [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
        maxlen=MAX_LEN,
        dtype="long",
        value=0.0,
        truncating="post",
        padding="post",
    )

    tags = pad_sequences(
        [[l for l in lab] for lab in labels],
        maxlen=MAX_LEN,
        value=tag2idx["PAD"],
        padding="post",
        dtype="long",
        truncating="post",
    )

    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, random_state=2018, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.LongTensor(tr_tags)
    val_tags = torch.LongTensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    model = BertForTokenClassification.from_pretrained(
        "bert-base-cased", num_labels=len(tag2idx), output_attentions=False, output_hidden_states=False
    )

    model = model.to(device)

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.01,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0},
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)

    epochs = 3
    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_values, validation_loss_values = [], []

    for _ in trange(epochs, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(tqdm(train_dataloader)):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [
            BIO_entities[p_i]
            for p, l in zip(predictions, true_labels)
            for p_i, l_i in zip(p, l)
            if BIO_entities[l_i] != "PAD"
        ]
        valid_tags = [BIO_entities[l_i] for l in true_labels for l_i in l if BIO_entities[l_i] != "PAD"]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Validation F1-Score: {}".format(f1_score([pred_tags], [valid_tags])))
        print()

    path_to_save_model = "util/random_scripts/data/model_bert_ner"
    model.save_pretrained(path_to_save_model)

    survey_tickets, entities_survey = load_survey_tickets_entities("ticket_generation/data/survey_tickets")

    output_file = []
    for ticket in survey_tickets:
        tokenized_sentence = tokenizer.encode(ticket)
        input_ids = torch.tensor([tokenized_sentence]).to(device)

        with torch.no_grad():
            output = model(input_ids)
        label_indices = np.argmax(output[0].to("cpu").numpy(), axis=2)

        # join bpe split tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids.to("cpu").numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(BIO_entities[label_idx])
                new_tokens.append(token)

        output_file.append({"tokens": new_tokens, "labels": new_labels})

        for token, label in zip(new_tokens, new_labels):
            print("{}\t{}".format(label, token))

    with open("util/random_scripts/data/output_test_bert_ner.json", "w") as f:
        json.dump(output_file, f)


if __name__ == "__main__":
    main()
