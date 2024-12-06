#!/usr/bin/env python
# coding=utf-8

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict
from sklearn.metrics import roc_auc_score

import numpy as np
from datasets import load_dataset
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# Ensure compatibility with required package versions
os.environ['CURL_CA_BUNDLE'] = ''
check_min_version("4.9.0")

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Data-related arguments for training, evaluation, and testing.
    Includes paths to data files, sequence lengths, and caching options.
    """
    task_name: Optional[str] = field(default=None, metadata={"help": "Name of the task to train on."})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name from the library."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "Dataset configuration name."})
    max_seq_length: int = field(default=2048, metadata={"help": "Maximum sequence length after tokenization."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite cached datasets."})
    pad_to_max_length: bool = field(default=True, metadata={"help": "Pad sequences to max length."})
    train_file: Optional[str] = field(default=None, metadata={"help": "CSV file for training data."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "CSV file for validation data."})
    test_file: Optional[str] = field(default=None, metadata={"help": "CSV file for testing data."})
    from_scratch: bool = field(default=False, metadata={"help": "Train from scratch without pretrained weights."})

@dataclass
class ModelArguments:
    """
    Arguments for configuring and loading the model and tokenizer.
    """
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or identifier."})
    config_name: Optional[str] = field(default=None, metadata={"help": "Config name or path."})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Tokenizer name or path."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Directory for caching models."})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Use fast tokenizer if available."})
    model_revision: str = field(default="main", metadata={"help": "Specific version of the model."})

class CustomTrainerCallback(TrainerCallback):
    """
    Custom callback for logging progress during training and evaluation.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero:
            logger.info(f"Step {state.global_step}, Epoch {state.epoch}: {logs}")

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"Epoch {int(state.epoch)} has ended.")

def main():
    # Argument parsing and setup
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%Y/%m/%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    logger.setLevel(training_args.get_process_log_level())
    logger.info(f"Training/evaluation parameters {training_args}")

    # Resume training if a checkpoint is found
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            logger.info(f"Resuming training from checkpoint {last_checkpoint}")

    # Set random seed
    set_seed(training_args.seed)

    # Load datasets
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file, "test": data_args.test_file}
    raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)

    # Prepare labels and tokenizer
    is_regression = False
    label_list = raw_datasets["train"].unique("label")
    label_list.sort()
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=num_labels, cache_dir=model_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir)

    # Define preprocessing function
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=data_args.max_seq_length, truncation=True)

    # Preprocess datasets
    raw_datasets = raw_datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    # Prepare datasets for training, evaluation, and prediction
    train_dataset = raw_datasets["train"] if training_args.do_train else None
    eval_dataset = raw_datasets["validation"] if training_args.do_eval else None
    test_dataset = raw_datasets["test"] if training_args.do_predict else None

    # Define metrics
    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        accuracy = accuracy_score(p.label_ids, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[CustomTrainerCallback()],
    )

    # Training loop
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        predictions, labels, _ = trainer.predict(test_dataset)
        auroc = roc_auc_score(labels, torch.softmax(torch.tensor(predictions), dim=1)[:, 1].numpy())
        logger.info(f"AUROC: {auroc:.4f}")

if __name__ == "__main__":
    main()
