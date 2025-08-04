# EDD

The official code for Expandable and Differentiable Dual Memory(EDD)




# CIFAR10
python main.py --model EDD --dataset seq-cifar10 --backbone edd --embedDim 200 --lambda_memory 2.0 --lambda_orthogonal 1.0 --memory_pruning_ratio 0.225 --ba_lr 0.0001 --ba_epochs 20 --lr 0.001
# CIFAR100
python main.py --model EDD --dataset seq-cifar100 --backbone edd --embedDim 1000 --lambda_memory 2.0 --lambda_orthogonal 1.0 --memory_pruning_ratio 0.225 --ba_lr 0.0001 --ba_epochs 20 --lr 0.001
# TinyImagenet
python main.py --model EDD --dataset seq-tinyimg --backbone edd --embedDim 1000 --lambda_memory 2.0 --lambda_orthogonal 1.0 --memory_pruning_ratio 0.225 --ba_lr 0.0001 --ba_epochs 20 --lr 0.001
