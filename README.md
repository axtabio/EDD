# EDD

The official code for Expandable and Differentiable Dual Memories with Orthogonal Regularization (EDD)

This code require Mammoth - A PyTorch Framework for Benchmarking Continual Learning package and enviroments

Usage Examples
## CIFAR10
python main.py --model EDD --dataset seq-cifar10 --backbone edd --embedDim 1000 --lambda_memory 20.0 --lambda_orthogonal 10.0 --memory_pruning_ratio 0.15 --ba_lr 0.0001 --ba_epochs 20 --lr 0.001
## CIFAR100
python main.py --model EDD --dataset seq-cifar100 --backbone edd --embedDim 1000 --lambda_memory 20.0 --lambda_orthogonal 10.0 --memory_pruning_ratio 0.15 --ba_lr 0.0001 --ba_epochs 20 --lr 0.001
## TinyImagenet
python main.py --model EDD --dataset seq-tinyimg --backbone edd --embedDim 1000 --lambda_memory 20.0 --lambda_orthogonal 10.0 --memory_pruning_ratio 0.15 --ba_lr 0.0001 --ba_epochs 20 --lr 0.001
