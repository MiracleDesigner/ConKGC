#!/usr/bin/env bash

data_dir="data/YAGO3-10"
model="conkgc"
group_examples_by_query="True"
use_action_space_bucketing="True"
add_reversed_training_edges="True"
entity_dim=200
relation_dim=200
bucket_interval=10
num_epochs=1000
num_wait_epochs=100
num_peek_epochs=2
batch_size=512
train_batch_size=512
dev_batch_size=128
learning_rate=0.001
grad_norm=0
emb_dropout_rate=0.3
relation_only="False"
margin=0.5
num_emb_hidden_layers=1
emb_hidden_size=250
num_hidden_layers=1
hidden_size=1024
use_tau="True"
tau=0.9