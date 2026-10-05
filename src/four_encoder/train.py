import argparse
import os
import pickle
import random

import numpy as np
import torch
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

from src.four_encoder.model import FourEncoderClassifier


LEARNING_RATE = 5e-6
EPOCHS = 5
WEIGHT_DECAY = 0.01
MAX_LENGTH = 512
NUM_LABELS = 3
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataframe(entries):
    rows = []

    for entry in entries:
        claim = str(entry["claim"])

        high = " ".join(
            entry.get("high", [])
        )

        medium = " ".join(
            entry.get("medium", [])
        )

        low = " ".join(
            entry.get("low", [])
        )

        background = str(
            entry.get("rag", "")
        )

        rows.append(
            {
                "text_high":
                    claim + " [SEP] " + high,

                "text_medium":
                    claim + " [SEP] " + medium,

                "text_low":
                    claim + " [SEP] " + low,

                "text_background":
                    claim + " [SEP] " + background,

                "labels":
                    int(entry["label"])
            }
        )

    return rows


def tokenize_dataset(dataset, tokenizer):

    def tokenize(batch):
        high = tokenizer(
            batch["text_high"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

        medium = tokenizer(
            batch["text_medium"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

        low = tokenizer(
            batch["text_low"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

        background = tokenizer(
            batch["text_background"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

        return {
            "input_ids_high":
                high["input_ids"],

            "attention_mask_high":
                high["attention_mask"],

            "input_ids_medium":
                medium["input_ids"],

            "attention_mask_medium":
                medium["attention_mask"],

            "input_ids_low":
                low["input_ids"],

            "attention_mask_low":
                low["attention_mask"],

            "input_ids_background":
                background["input_ids"],

            "attention_mask_background":
                background["attention_mask"],

            "labels":
                batch["labels"]
        }

    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names
    )

    dataset.set_format("torch")

    return dataset


def compute_weights(train_rows):
    labels = [
        row["labels"]
        for row in train_rows
    ]

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=labels
    )

    return torch.tensor(
        weights,
        dtype=torch.float
    )


def train_fold(
    fold_id,
    folds_file,
    model_name,
    output_dir
):
    set_seed(SEED)

    with open(
        folds_file,
        "rb"
    ) as file:
        folds = pickle.load(file)

    train_entries, eval_entries = folds[fold_id]

    train_rows = build_dataframe(
        train_entries
    )

    eval_rows = build_dataframe(
        eval_entries
    )

    train_dataset = Dataset.from_list(
        train_rows
    )

    eval_dataset = Dataset.from_list(
        eval_rows
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer
    )

    eval_dataset = tokenize_dataset(
        eval_dataset,
        tokenizer
    )

    class_weights = compute_weights(
        train_rows
    )

    model = FourEncoderClassifier(
        model_name=model_name,
        num_labels=NUM_LABELS,
        class_weights=class_weights
    )

    fold_output = os.path.join(
        output_dir,
        f"fold_{fold_id}"
    )

    os.makedirs(
        fold_output,
        exist_ok=True
    )

    training_args = TrainingArguments(
        output_dir=fold_output,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=EPOCHS,
        fp16=torch.cuda.is_available(),
        logging_strategy="epoch",
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()

    trainer.save_model(
        os.path.join(
            fold_output,
            "final_model"
        )
    )

    tokenizer.save_pretrained(
        os.path.join(
            fold_output,
            "final_model"
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        required=True
    )

    parser.add_argument(
        "--folds",
        required=True
    )

    parser.add_argument(
        "--model",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    train_fold(
        fold_id=args.fold,
        folds_file=args.folds,
        model_name=args.model,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()