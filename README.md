# Feedback-Gradient-Descent
For CIFAR-10:

    python main.py --model resnet --depth 28 --width 10 --optim_method FGD --lr 0.05 -lrg 0.2 --feedback 0.4 --stiefel 1 --dataset CIFAR10 --gpu_id [gpu_id] --save [log_path]

For CIFAR-100:

    python main.py --model resnet --depth 28 --width 10 --optim_method FGD --lr 0.08 -lrg 0.16 --feedback 0.4 --stiefel 1 --dataset CIFAR100 --gpu_id [gpu_id] --save [log_path]
