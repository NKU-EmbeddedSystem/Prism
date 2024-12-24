#!/bin/bash

#1. prepare data 
data_dir=/home/cy/ccd/Bert-Fine-Tune/datapreprocess/datasets/test
task_name=MRPC

pretrain_dir=/home/cy/ccd/Bert-Fine-Tune/pretrained_models/
vocab_fpath=${pretrain_dir}/vocab.txt

output_dir=./finetune/
options="--data_dir $data_dir/$task_name"
options="$options --task_name $task_name "

options="$options --bert_model bert-base-uncased "
options="$options --do_train "
options="$options --do_eval "
options="$options --pretrain_dir $pretrain_dir "
options="$options --vocab_fpath $vocab_fpath "
options="$options --output_dir $output_dir/$task_name/train_epoch_2000-weight-gcjtest"
options="$options --do_lower_case"
options="$options --local_rank 5"
options="$options --learning_rate 1e-5"
options="$options --train_batch_size 32"
options="$options --max_seq_length 64"
options="$options --num_train_epochs 2000"
options="$options --num_labels 2"
options="$options --emb_size 96"

#2. fine-tune it
python ./seq_classify.py $options