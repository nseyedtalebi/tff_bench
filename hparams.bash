#!/usr/bin/env bash

# Cross-task hyperparameters
export CROSS_TASK_HPARAMS="clients_per_round client_epochs client_datasets_seed total_rounds client_batch_size"
export CIFAR_100="10 1 1 4000 20"
export EMNIST_AE="10 1 1 3000 20"
export EMNIST_CR="10 1 1 1500 20"
export SHAKESPEARE="10 1 1 1200 4"
export STACKOVERFLOW_LR="10 1 1 1500 100"
export STACKOVERFLOW_NWP="50 1 1 1500 16"

# Experiment-specific hyperparameters
export CIFAR_PARAMS="cifar100_crop_size 24"
export EMNIST_CR_PARAMS="emnist_cr_model cnn"
export SHAKESPEARE_PARAMS="shakespeare_sequence_length 80"
export STACKOVERFLOW_LR_PARAMS="so_lr_vocab_tokens_size 10000 so_lr_vocab_tags_size 500 so_lr_num_validation_examples 10000 so_lr_max_elements_per_user 1000"
export STACKOVERFLOW_NWP_PARAMS="so_nwp_vocab_size 10000 so_nwp_num_oov_buckets so_nwp_sequence_length 20 so_nwp_num_validation_examples 10000 so_nwp_max_elements_per_user 1000 so_nwp_embedding_size 96 so_nwp_latent_size 670 so_nwp_num_layers 1"

# Best optimizer for each task according to experiments by Google Research
export BEST_OPT_CIFAR_100="FedYogi"
export BEST_OPT_EMNIST_AE="FedYogi"
export BEST_OPT_EMNIST_CR="FedAdam FedYogi FedAvgM"
export BEST_OPT_SHAKESPEARE="FedAdagrad FedYogi FedAvgM"
export BEST_OPT_STACKOVERFLOW_LR="FedAdagrad"
export BEST_OPT_STACKOVERFLOW_NWP="FedAdam FedYogi"

# Optimizer config
export SERV_FEDAVG="server_optimizer sgd server_sgd_momentum 0.0"
export SERV_FEDAVGM="server_optimizer sgd server_sgd_momentum 0.9"
export SERV_FEDADAGRAD="server_optimizer adagrad server_adagrad_initial_accumulator_value 0.0 server_adagrad_epsilon 0.001"
export SERV_FEDADAM="server_optimizer adam server_adam_epsilon 0.001"
export SERV_FEDYOGI="server_optimizer yogi server_yogi_initial_accumulator_value 0.0 server_yogi_epsilon 0.001" # found to maximize pic-a-nic baskets!
export CLIENT_SGD="client_optimizer sgd"

# Client learning rates
export CL_LR_CIFAR_100="-1 -1.5 -1.5 -1.5 -1"
export CL_LR_EMNIST_AE="1.5 1 1 0.5 1"
export CL_LR_EMNIST_CR="-1.5 -1.5 -1.5 -1.5 -1"
export CL_LR_SHAKESPEARE="0 0 0 0 0"
export CL_LR_STACKOVERFLOW_LR="2 2 2 2 2"
export CL_LR_STACKOVERFLOW_NWP="-0.5 -0.5 -0.5 -0.5 -0.5"

# Server learning rates (uses $OPTIMIZERS for headers)
export SRV_LR_CIFAR_100="-1 0 0 0 0.5"
export SRV_LR_EMNIST_AE="-1.5 -1.5 -1.5 0 0"
export SRV_LR_EMNIST_CR="-1 -2.5 -2.5 -0.5 0"
export SRV_LR_SHAKESPEARE="-0.5 -2 -2 -0.5 0"
export SRV_LR_STACKOVERFLOW_LR="1 -0.5 -0.5 0 0"
export SRV_LR_STACKOVERFLOW_NWP="-1.5 -2 -2 0 0"

# Optimizers for use with learning rates above
export LR_OPTS="Adagrad Adam Yogi AvgM Avg"

# Epsilon values
export OPTS_WITH_EPS="Adagrad Adam Yogi"
export EPS_CIFAR_100="-2 -1 -1"
export EPS_EMNIST_AE="-3 -3 -3"
export EPS_EMNIST_CR="-2 -4 -4"
export EPS_SHAKESPEARE="-1 -3 -3"
export EPS_STACKOVERFLOW_LR="-2 -5 -5"
export EPS_STACKOVERFLOW_NWP="-4 -5 -5"