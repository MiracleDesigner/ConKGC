#!/bin/bash

export PYTHONPATH=`pwd`
echo $PYTHONPATH

source $1
exp=$2
gpu=$3
ARGS=${@:4}

relation_only_flag=''
if [[ $relation_only = *"True"* ]]; then
    relation_only_flag="--relation_only"
fi
group_examples_by_query_flag=''
if [[ $group_examples_by_query = *"True"* ]]; then
    group_examples_by_query_flag="--group_examples_by_query"
fi
use_action_space_bucketing_flag=''
if [[ $use_action_space_bucketing = *"True"* ]]; then
    use_action_space_bucketing_flag='--use_action_space_bucketing'
fi
add_reversed_training_edges_flag=''
if [[ $add_reversed_training_edges = *"True"* ]]; then
    add_reversed_training_edges_flag="--add_reversed_training_edges"
fi
use_tau_flag=''
if [[ $use_tau = *"True"* ]]; then
    use_tau_flag="--use_tau"
fi

cmd="python3 -m src.experiments \
    --data_dir $data_dir \
    $exp \
    --model $model \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --bucket_interval $bucket_interval \
    --num_epochs $num_epochs \
    --num_wait_epochs $num_wait_epochs \
    --num_peek_epochs $num_peek_epochs \
    --batch_size $batch_size \
    --train_batch_size $train_batch_size \
    --dev_batch_size $dev_batch_size \
    --margin $margin \
    --learning_rate $learning_rate \
    --grad_norm $grad_norm \
    --emb_dropout_rate $emb_dropout_rate \
    --num_emb_hidden_layers $num_emb_hidden_layers \
    --emb_hidden_size $emb_hidden_size \
    --num_hidden_layers $num_hidden_layers \
    --hidden_size $hidden_size \
    --tau $tau \
    $relation_only_flag \
    $group_examples_by_query_flag \
    $use_action_space_bucketing_flag \
    $add_reversed_training_edges_flag \
    $use_tau_flag \
    --gpu $gpu \
    $ARGS"

echo "Executing $cmd"

$cmd
