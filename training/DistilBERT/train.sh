                                                                                                  train.sh
#!/bin/bash

# Define variables for model, dataset, and output paths
plm_dir="distilbert-base-uncased"
seed=42629309
data_path="/path/to/data"  #Change this path
train_file="$data_path/train.csv"
valid_file="$data_path/valid.csv"
test_file="$data_path/test.csv"
out_dir="/output/path/output_samples_${seed}_DistilBERT"   #Change this path
time=$(date +'%Y%m%d_%H%M%S')
mkdir -p $out_dir

# Command to run the training script using specified configurations
CUDA_VISIBLE_DEVICES=0 python3 /path/to/main.py \    #Change this path
  --do_train \
  --do_eval \
  --do_predict \
  --model_name_or_path $plm_dir \
  --train_file $train_file \
  --validation_file $valid_file \
  --test_file $test_file \
  --max_seq_length 256 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --evaluation_strategy "epoch" \
  --logging_steps 100 \
  --overwrite_output_dir \
  --fp16 \
  --seed $seed \
  --output_dir $out_dir 2>&1 | tee $out_dir/log.train.$time
