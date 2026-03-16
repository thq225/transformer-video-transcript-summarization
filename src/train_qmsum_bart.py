import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# =========================
# Config
# =========================
# MODEL_NAME = "facebook/bart-large-cnn"
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
DATA_DIR = Path("data/qmsum")
OUTPUT_DIR = "outputs/qmsum_bart_baseline"

TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

MAX_SOURCE_LENGTH = 256 # 1024
MAX_TARGET_LENGTH = 64 # 128

TRAIN_BATCH_SIZE = 1 #2
EVAL_BATCH_SIZE = 1 #2
GRAD_ACCUM_STEPS = 1 #8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3 #3
WEIGHT_DECAY = 0.01
SEED = 123


# MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
# TRAIN_BATCH_SIZE = 1
# EVAL_BATCH_SIZE = 1
# GRAD_ACCUM_STEPS = 1
# NUM_EPOCHS = 1


def load_qmsum_jsonl(file_path: Path) -> List[Dict]:
    """
    Load one QMSum split from a .jsonl file.

    Each line is one meeting record, containing:
      - meeting_transcripts
      - general_query_list
      - specific_query_list

    We flatten each meeting into many training examples:
      {
        "meeting_id": ...,
        "query": ...,
        "transcript": ...,
        "summary": ...
      }
    """
    examples = []

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path.resolve()}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            # create a stable meeting id
            meeting_id = item.get("meeting_id", f"{file_path.stem}_{line_idx}")

            # build transcript string
            transcript_turns = []
            for turn in item.get("meeting_transcripts", []):
                speaker = turn.get("speaker", "Speaker")
                text = turn.get("content", "").strip()
                if text:
                    transcript_turns.append(f"{speaker}: {text}")

            transcript = "\n".join(transcript_turns).strip()

            if not transcript:
                continue

            # general queries
            for q in item.get("general_query_list", []):
                query = q.get("query", "").strip()
                answer = q.get("answer", "").strip()

                if query and answer:
                    examples.append(
                        {
                            "meeting_id": meeting_id,
                            "query": query,
                            "transcript": transcript,
                            "summary": answer,
                        }
                    )

            # specific queries
            for q in item.get("specific_query_list", []):
                query = q.get("query", "").strip()
                answer = q.get("answer", "").strip()

                if query and answer:
                    examples.append(
                        {
                            "meeting_id": meeting_id,
                            "query": query,
                            "transcript": transcript,
                            "summary": answer,
                        }
                    )

    return examples


def load_qmsum_dataset() -> DatasetDict:
    """
    Load official QMSum train/val/test splits from jsonl files.
    """
    train_examples = load_qmsum_jsonl(TRAIN_FILE)
    val_examples = load_qmsum_jsonl(VAL_FILE)
    test_examples = load_qmsum_jsonl(TEST_FILE)

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_examples),
            "validation": Dataset.from_list(val_examples),
            "test": Dataset.from_list(test_examples),
        }
    )

    # TODO:
    # dataset["train"] = dataset["train"].select(range(20))
    # dataset["validation"] = dataset["validation"].select(range(5))
    # dataset["test"] = dataset["test"].select(range(5))

    return dataset


def preprocess_function(batch, tokenizer):
    """
    Format input as:
      query: ...
      transcript:
      ...

    Then tokenize inputs and target summaries.
    """
    inputs = [
        f"query: {q}\n\ntranscript:\n{t}"
        for q, t in zip(batch["query"], batch["transcript"])
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        text_target=batch["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels


def main():
    set_seed(SEED)

    print("Loading QMSum dataset...")
    dataset = load_qmsum_dataset()
    print(dataset)
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    print(f"Test examples: {len(dataset['test'])}")

    tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    tokenized_dataset = dataset.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # predictions may come as tuple in some HF versions
        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = float(np.mean(prediction_lens))

        return {k: round(v, 4) for k, v in result.items()}

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        fp16=False,
        logging_steps=10,
        logging_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["test"],
        metric_key_prefix="test",
    )
    print(test_metrics)

    print("Saving model...")
    save_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save a few predictions for inspection
    print("Generating sample predictions...")
    n_samples = min(10, len(tokenized_dataset["test"]))

    preds_output = trainer.predict(tokenized_dataset["test"].select(range(n_samples)))
    pred_ids = preds_output.predictions
    label_ids = preds_output.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    raw_test_subset = dataset["test"].select(range(n_samples))

    out_path = Path(OUTPUT_DIR) / "sample_predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex, pred, label in zip(raw_test_subset, decoded_preds, decoded_labels):
            row = {
                "meeting_id": ex["meeting_id"],
                "query": ex["query"],
                "reference_summary": label.strip(),
                "generated_summary": pred.strip(),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved sample predictions to {out_path}")


if __name__ == "__main__":
    main()
