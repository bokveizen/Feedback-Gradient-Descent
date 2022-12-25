# Feedback-Gradient-Descent

Bu, Fanchen and Dong Eui Chang. “Feedback Gradient Descent: Efficient and Stable Optimization with Orthogonality for DNNs.” AAAI (2022).

    @inproceedings{Bu2022FeedbackGD,
      title={Feedback Gradient Descent: Efficient and Stable Optimization with Orthogonality for DNNs},
      author={Fanchen Bu and Dong Eui Chang},
      booktitle={AAAI},
      year={2022}
    }

For CIFAR-10:

    python main.py --model resnet --depth 28 --width 10 --optim_method FGD --lr 0.05 -lrg 0.2 --feedback 0.4 --stiefel 1 --dataset CIFAR10 --gpu_id [gpu_id] --save [log_path]

For CIFAR-100:

    python main.py --model resnet --depth 28 --width 10 --optim_method FGD --lr 0.08 -lrg 0.16 --feedback 0.4 --stiefel 1 --dataset CIFAR100 --gpu_id [gpu_id] --save [log_path]
