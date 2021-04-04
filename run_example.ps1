$params = @{
    'task'='cifar100'
    'total_rounds'=100
    'client_optimizer'='sgd'
    'client_learning_rate'=0.1
    'client_batch_size'=20
    'server_optimizer'='sgd'
    'server_learning_rate'=1.0
    'clients_per_round'=10
    'client_epochs_per_round'=1
    'experiment_name'='cifar100_no_dp'
}
docker run -it -v C:\Users\NiMo3\Documents\projects\tff_bench:/tff tff python main.py @params
#\
#--clip=0.05 --noise_multiplier=1.5 --adaptive_clip_learning_rate=0 \
#--target_unclipped_quantile=0.5 --clipped_count_budget_allocation=0.1