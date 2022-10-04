# On-attacking-Out-domain-Uncertainty-Estimation-in-Deep-Neural-Networks

This is the official code for the [IJCAI'22 paper](https://www.ijcai.org/proceedings/2022/0678.pdf) "On Attacking Out-Domain Uncertainty Estimation in Deep Neural Networks".

<img src=pics/intro.png>

In many applications with real-world consequences, it is crucial to develop reliable uncertainty estimation for the predictions made by the AI decision systems. Targeting at the goal of estimating uncertainty, various deep neural network (DNN) based uncertainty estimation algorithms have been proposed. However, the robustness of the uncertainty returned by these algorithms has not been systematically explored. In this work, to raise the awareness of the research community on robust uncertainty estimation, we show that state-of-the-art uncertainty estimation algorithms could fail catastrophically under our proposed adversarial attack despite their impressive performance on uncertainty estimation. In particular, we aim at attacking the out-domain uncertainty estimation: under our attack, the uncertainty model would be fooled to make high-confident predictions for the out-domain data, which they originally would have rejected. Extensive experimental results on various benchmark image datasets show that the uncertainty estimated by state-of-the-art methods could be easily corrupted by our attack.

## Citing 

Please consider citing the following paper if you use our methods in your research:
```
@inproceedings{zeng2022attacking,
  title={On Attacking Out-Domain Uncertainty Estimation in Deep Neural Networks},
  author={Zeng, Huimin and Yue, Zhenrui and Zhang, Yang and Kou, Ziyi and Shang, Lanyu and Wang, Dong},
  year={2022},
  organization={IJCAI}
}
```

## Requirements

For our running environment see requirements.txt

## Data folder and dataloader
- MNIST, CIFAR10, SVHN loaded from torchvision.datasets
- NotMNIST downloaded from [here](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html)
   - after downloaded, the data file should be placed in the same folder of MNISTT, CIFAR10, SVHN.
- Example folder structure
```
    ├── ...
    ├── Data                   
    │   ├── MNIST 
    │   │   └── raw
    │   ├── notMNIST_small 
    │   │   ├── A
    │   │   └── ...
    │   ├── cifar-10-batches-py
    │   └── SVHN
    └── Code
        ├── dataloader.py
        ├── train_baseline.py
        └── ...
```
## Scripts for training and attacking various baseline uncertainty estimation models.

- Deep Ensemble and Adversarially Trained Deep Ensemble [paper link](https://proceedings.neurips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf)
   - Training Example
       ```
       python train_ensemble.py --model 'LeNet' --seed X                      
       
       python train_ensemble_at.py --model 'LeNet' --epsilon 0.1 --seed X    # epsilon is the adversarial radius used for training
       ```
   - Evaluation Example
      ```
      # evaluate baseline models
      python eval_model --model 'LeNet' --mode 'deep_ensemble' --num_ensembles 10 --epsilon_attack 0.1
      python eval_model --model 'LeNet' --mode 'deep_ensemble_at' --num_ensembles 10 --epsilon 0.1 --epsilon_attack 0.1
      
      # launch out-domain adversarial attacks
      python eval_model --model 'LeNet' --mode 'attack_deep_ensemble' --num_ensembles 10 --epsilon_attack 0.1
      python eval_model --model 'LeNet' --mode 'attack_deep_ensemble_at' --num_ensembles 10 --epsilon 0.1 --epsilon_attack 0.1
      ```
   - Hyperparameters
      ```
      --model           # 'LeNet' for MNIST vs NotMNIST; 'ResNet-18' for CIFAR10 vs SVHN
      --seed            # X = [0,1,...9] for 10-ensemble model
      --epsilon         # adversarial radius used for adversarial training
      --epsilon_attack  # adversarial radius used for evaluation
      --mode            # evaluation mode, either for evaluating baseline model or launching attack
      ```
      
- Deterministic Uncertainty Quantification [paper link](https://arxiv.org/pdf/2003.02037.pdf)
   - Training Example (code adapted from [here](https://github.com/y0ast/deterministic-uncertainty-quantification))
      ```
      python train_duq.py --model 'LeNet_DUQ' --lambda_reg 0.1 
      ```
   - Evaluation Example
      ```
      # evaluate baseline models
      python eval_model --model 'LeNet_DUQ' --mode 'duq' --lambda_reg 0.1 --epsilon_attack 0.1
      
      # launch out-domain adversarial attacks
      python eval_model --model 'LeNet_DUQ' --mode 'attack_duq' --lambda_reg 0.1 --epsilon_attack 0.1
      ```
   - Hyperparameters
      ```
      --model        # 'LeNet_DUQ' for MNIST vs NotMNIST; 'ResNet-18_DUQ' for CIFAR10 vs SVHN
      --lambda_reg   # penalty strength
      --mode         # evaluation mode, either for evaluating baseline model or launching attack
      ```
      
- Deterministic Uncertainty Estimation [paper link](https://arxiv.org/pdf/2102.11409.pdf)
   - Training Example (code adapted from [here](https://github.com/y0ast/DUE))
      ```
      python train_due.py --model 'LeNet_DUE'
      ```
   - Evaluation Example
      ```
      # evaluate baseline models
      python eval_model --model 'LeNet_DUE' --mode 'due' --epsilon_attack 0.1
      
      # launch out-domain adversarial attacks
      python eval_model --model 'LeNet_DUE' --mode 'attack_due' --epsilon_attack 0.1
      ```
   - Hyperparameters
      ```
      --model  # 'LeNet_DUE' for MNIST vs NotMNIST; 'ResNet-18_DUE' for CIFAR10 vs SVHN
      --mode   # evaluation mode, either for evaluating baseline model or launching attack
      ```
   - Note: https://github.com/cornellius-gp/gpytorch
- Spectral-Normalized Gaussian Process [paper link](https://papers.nips.cc/paper/2020/file/543e83748234f7cbab21aa0ade66565f-Paper.pdf)
   - Training Example (code adapted from [here](https://github.com/y0ast/DUE))
      ```
      python train_sngp.py --model 'LeNet_SNGP'
      ```
   - Evaluation Example
      ```
      # evaluate baseline models
      python eval_model --model 'LeNet_SNGP' --mode 'sngp' --epsilon_attack 0.1
      
      # launch out-domain adversarial attacks
      python eval_model --model 'LeNet_SNGP' --mode 'attack_sngp' --epsilon_attack 0.1
      ```
   - Hyperparameters
      ```
      --model  # 'LeNet_SNGP' for MNIST vs NotMNIST; 'ResNet-18_SNGP' for CIFAR10 vs SVHN
      --mode   # evaluation mode, either for evaluating baseline model or launching attack
      ```
   - Note: [`gpytorch`](https://github.com/cornellius-gp/gpytorch) required
      
## Performance

We evaluate the efficacy of our proposed out-domain uncertainty attack by assessing to which extent, the victim model could be deceived to make high-confident predictions for perturbed out-domain data.

### Main Results

<img src=pics/main_results.png width=500>

### Robustness Study

<img src=pics/robustness_study.png width=500>

### Concrete Examples

<img src=pics/exp_1.png width=500>
<img src=pics/exp_2.png width=500>


## Acknowledgement

During the implementation we base our code mostly on the repos of baseline uncertainty estimation methods as cited above. Many thanks to these authors for their great work!