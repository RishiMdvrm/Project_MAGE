#!/usr/bin/env python
# coding=utf-8

# Import necessary libraries for logging, dataset handling, model training, and evaluation
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np
from datasets import load_dataset
import torch
from torch.utils.tensorboard import SummaryWriter

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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Ensures compatibility with minimum versions
os.environ['CURL_CA_BUNDLE'] = ''
check_min_version("4.9.0")

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments for data input and preprocessing.
    """
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "Task name (if applicable)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the training data file (CSV)."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the validation data file (CSV)."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the test data file (CSV)."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for tokenization."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite cached datasets."}
    )

@dataclass
class ModelArguments:
    """
    Arguments for model configuration and loading.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to the pretrained model or model identifier from HuggingFace."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path."}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path."}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use a fast tokenizer."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to store cached models and tokenizers."}
    )

class CustomTrainerCallback(TrainerCallback):
    """
    Custom callback for monitoring training progress.
    Logs metrics and events during and at the end of each epoch.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero:
            logger.info(f"Step {state.global_step}, Epoch {state.epoch}: {logs}")

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"Epoch {int(state.epoch)} has ended.")

def main():
    # Parse arguments for model, data, and training
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load arguments from a JSON file
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Parse command-line arguments
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set up logging and log configuration
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(training_args.get_process_log_level())
    logger.info(f"Training/evaluation parameters {training_args}")

    # Resume training from the last checkpoint if applicable
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            logger.info(f"Resuming training from checkpoint {last_checkpoint}")
        else:
            logger.info("Starting training from scratch.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load datasets for train, validation, and test
    data_files = {}
    if training_args.do_train:
        data_files["train"] = data_args.train_file
    if training_args.do_eval:
        data_files["validation"] = data_args.validation_file
    if training_args.do_predict:
        data_files["test"] = data_args.test_file

    raw_datasets = load_dataset(
        "csv",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )

    # Determine the number of labels (binary classification default)
    is_regression = False
    num_labels = len(raw_datasets["train"].unique("label")) if training_args.do_train else 2

    # Load model, tokenizer, and configuration
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Preprocess datasets for tokenization and alignment with model input
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=data_args.max_seq_length, truncation=True)

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    # Split datasets into train, eval, and test
    train_dataset = raw_datasets["train"] if training_args.do_train else None
    eval_dataset = raw_datasets["validation"] if training_args.do_eval else None
    test_dataset = raw_datasets["test"] if training_args.do_predict else None

    # Define metrics for evaluation
    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[CustomTrainerCallback()],
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()
        trainer.save_metrics("train", train_result.metrics)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.save_metrics("eval", metrics)

    # Prediction and saving results
    if training_args.do_predict:
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        trainer.save_metrics("predict", metrics)
        preds = np.argmax(predictions, axis=1)
        with open(os.path.join(training_args.output_dir, "test_predictions.txt"), "w") as writer:
            for index, pred in enumerate(preds):
                writer.write(f"{index}\t{pred}\n")

if __name__ == "__main__":
    main()
