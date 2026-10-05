import argparse
import json
import os
import pickle

import numpy as np
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

from src.four_encoder.model import FourEncoderClassifier


MAX_LENGTH = 512
NUM_LABELS = 3


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


def evaluate_fold(
    fold_id,
    folds_file,
    model_path,
    output_dir
):
    with open(
        folds_file,
        "rb"
    ) as file:
        folds = pickle.load(file)

    _, eval_entries = folds[fold_id]

    eval_rows = build_dataframe(
        eval_entries
    )

    eval_dataset = Dataset.from_list(
        eval_rows
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    eval_dataset = tokenize_dataset(
        eval_dataset,
        tokenizer
    )

    model = FourEncoderClassifier(
        model_name=model_path,
        num_labels=NUM_LABELS
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=1,
            report_to="none",
            remove_unused_columns=False
        )
    )

    predictions = trainer.predict(
        eval_dataset
    )

    logits = predictions.predictions

    if isinstance(
        logits,
        tuple
    ):
        logits = logits[0]

    y_pred = np.argmax(
        logits,
        axis=1
    )

    y_true = predictions.label_ids

    report = classification_report(
        y_true,
        y_pred,
        digits=4
    )

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    np.save(
        os.path.join(
            output_dir,
            "y_true.npy"
        ),
        y_true
    )

    np.save(
        os.path.join(
            output_dir,
            "y_pred.npy"
        ),
        y_pred
    )

    with open(
        os.path.join(
            output_dir,
            "classification_report.txt"
        ),
        "w"
    ) as file:
        file.write(report)
        file.write(
            "\n\nConfusion Matrix:\n"
        )
        file.write(str(matrix))

    with open(
        os.path.join(
            output_dir,
            "metrics.json"
        ),
        "w"
    ) as file:
        json.dump(
            {
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1)
            },
            file,
            indent=4
        )

    print(report)
    print("\nConfusion Matrix:")
    print(matrix)
    print("\nAccuracy:", accuracy)
    print("Macro F1:", macro_f1)


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

    evaluate_fold(
        fold_id=args.fold,
        folds_file=args.folds,
        model_path=args.model,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()