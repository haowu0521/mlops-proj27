import argparse
import os
import time
from typing import Dict, Any, List

import mlflow
import numpy as np
import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import evaluate


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_demo_dataset() -> DatasetDict:
    train_examples: List[Dict[str, str]] = [
        {
            "transcript": "Alice: We need to finish the UI this week. Bob: I will handle the backend API. Carol: Let's meet again on Friday.",
            "summary": "The team discussed finishing the UI, assigned the backend API to Bob, and scheduled another meeting on Friday."
        },
        {
            "transcript": "Tom: The server was unstable yesterday. Jerry: I will check the logs. Anna: We should restart the deployment tonight.",
            "summary": "The team discussed server instability, assigned log checking to Jerry, and planned a restart for tonight."
        },
        {
            "transcript": "Mia: We still need slides for the demo. Leo: I can prepare the architecture page. Nina: I will polish the conclusion slide.",
            "summary": "The team discussed unfinished demo slides, assigned the architecture page to Leo, and the conclusion slide to Nina."
        },
        {
            "transcript": "Sara: Users said the summary was too long. Kevin: We should shorten the output. Emma: Let's add action items separately.",
            "summary": "The team reviewed user feedback, decided to shorten summaries, and separate action items from the main summary."
        },
        {
            "transcript": "Ben: Our dataset has duplicates. Lisa: I will clean them tonight. Mark: Please freeze the split after cleaning.",
            "summary": "The team found duplicate data, assigned cleanup to Lisa, and agreed to freeze the split afterward."
        },
        {
            "transcript": "Ivy: The meeting notes missed two decisions. Owen: We need better evaluation. Chloe: Let's add ROUGE and manual review.",
            "summary": "The team discussed missing decisions in notes and agreed to improve evaluation with ROUGE and manual review."
        },
    ]

    val_examples: List[Dict[str, str]] = [
        {
            "transcript": "Alex: We need a Dockerfile. Sam: I will write it tonight. Dana: Please test it tomorrow morning.",
            "summary": "The team agreed to create a Dockerfile, assigned it to Sam, and planned testing for tomorrow morning."
        },
        {
            "transcript": "Paul: Training is slow on CPU. Jane: We should switch to GPU. Eric: Let's keep batch size small first.",
            "summary": "The team discussed slow CPU training, suggested moving to GPU, and decided to keep batch size small initially."
        },
    ]

    test_examples: List[Dict[str, str]] = [
        {
            "transcript": "Liam: The transcript quality is acceptable. Ava: Then we can focus on summarization first. Noah: ASR retraining can wait.",
            "summary": "The team agreed transcript quality was acceptable and decided to prioritize summarization before ASR retraining."
        },
        {
            "transcript": "Grace: We need one baseline and two candidates. Henry: Let's start simple. Ella: Record everything in MLflow.",
            "summary": "The team planned one baseline and two candidates, chose a simple starting point, and agreed to track runs in MLflow."
        },
    ]

    return DatasetDict(
        {
            "train": Dataset.from_list(train_examples),
            "validation": Dataset.from_list(val_examples),
            "test": Dataset.from_list(test_examples),
        }
    )


def load_data(cfg: Dict[str, Any]) -> DatasetDict:
    data_cfg = cfg["data"]
    train_file = data_cfg.get("train_file", "").strip()
    val_file = data_cfg.get("val_file", "").strip()
    test_file = data_cfg.get("test_file", "").strip()

    if train_file and val_file and test_file:
        data_files = {
            "train": train_file,
            "validation": val_file,
            "test": test_file,
        }
        dataset = load_dataset("json", data_files=data_files)
    else:
        print("No local dataset files provided. Using tiny built-in demo dataset.")
        dataset = build_demo_dataset()

    max_train = data_cfg.get("max_train_samples")
    max_val = data_cfg.get("max_val_samples")
    max_test = data_cfg.get("max_test_samples")

    if max_train:
        dataset["train"] = dataset["train"].select(range(min(max_train, len(dataset["train"]))))
    if max_val:
        dataset["validation"] = dataset["validation"].select(range(min(max_val, len(dataset["validation"]))))
    if max_test:
        dataset["test"] = dataset["test"].select(range(min(max_test, len(dataset["test"]))))

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    dataset = load_data(cfg)

    model_name = cfg["model"]["model_name"]
    text_column = cfg["data"]["text_column"]
    summary_column = cfg["data"]["summary_column"]
    max_input_length = cfg["model"]["max_input_length"]
    max_target_length = cfg["model"]["max_target_length"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    rouge = evaluate.load("rouge")

    def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        inputs = ["summarize meeting: " + x for x in examples[text_column]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
        )

        labels = tokenizer(
            text_target=examples[summary_column],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        result = {k: round(v * 100, 4) for k, v in result.items()}
        return result

    train_cfg = cfg["train"]
    evaluation_strategy = train_cfg.get("evaluation_strategy", train_cfg.get("eval_strategy", "epoch"))

    training_args = Seq2SeqTrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        weight_decay=float(train_cfg["weight_decay"]),
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=evaluation_strategy,
        save_strategy=train_cfg["save_strategy"],
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=train_cfg["seed"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name=cfg["run"]["candidate_name"]):
        mlflow.log_param("candidate_name", cfg["run"]["candidate_name"])
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset_name", cfg["data"]["dataset_name"])
        mlflow.log_param("max_train_samples", cfg["data"]["max_train_samples"])
        mlflow.log_param("max_val_samples", cfg["data"]["max_val_samples"])
        mlflow.log_param("max_test_samples", cfg["data"]["max_test_samples"])
        mlflow.log_param("learning_rate", train_cfg["learning_rate"])
        mlflow.log_param("num_train_epochs", train_cfg["num_train_epochs"])
        mlflow.log_param("train_batch_size", train_cfg["per_device_train_batch_size"])
        mlflow.log_param("eval_batch_size", train_cfg["per_device_eval_batch_size"])
        mlflow.log_param("max_input_length", max_input_length)
        mlflow.log_param("max_target_length", max_target_length)

        if torch.cuda.is_available():
            mlflow.log_param("device", "cuda")
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
        else:
            mlflow.log_param("device", "cpu")
            mlflow.log_param("gpu_name", "none")

        start_time = time.time()

        train_result = trainer.train()
        train_metrics = train_result.metrics
        trainer.save_model(train_cfg["output_dir"])
        tokenizer.save_pretrained(train_cfg["output_dir"])

        eval_metrics = trainer.evaluate(
            max_length=max_target_length,
            num_beams=4,
        )

        total_train_time = time.time() - start_time

        mlflow.log_metric("train_runtime_sec", round(total_train_time, 4))

        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"train_{key}", float(value))

        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"eval_{key}", float(value))

        #mlflow.log_artifacts(train_cfg["output_dir"], artifact_path="model_output")

    print("Training finished.")
    print(f"MLflow experiment: {cfg['mlflow']['experiment_name']}")
    print(f"Run name: {cfg['run']['candidate_name']}")


if __name__ == "__main__":
    main()
