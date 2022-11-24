# DRA-BlackBoxAttack

An unofficial implementation of the paper《[Towards Understanding and Boosting Adversarial Transferability from a Distribution Perspective](https://arxiv.org/abs/2210.04213)》

![avatar](/data/target.png)



## Requirement  and install

- Clone this repository. Assume this repositry is downloaded to .`/DRA-BlackBoxATTACK/`

- Install dependencies

  - `cd DRA-BlackBoxATTACK`
  - `pip install -r requirements.txt`

  

## Prepare Dataset

- please download the dataset from the following link and extract images to the path “./data/ImageNet-10/" [imagenet10 | Kaggle](https://www.kaggle.com/datasets/liusha249/imagenet10)



## Usage

- first,please change project path `config/baseconfig.py  BaseConfig root and data_root` to your path

- second run main.py

  ```python
  python main.py  
  ```

- if choice wholeRunner, it contains a comparison with PGD
- you can change config in `config/config.py`



## Attack effect

- 原始模型resnet18 （Acc: 0.99) 
- 经过DRA_loss 微调过的 resnet18_DRA (Acc: 0.9564) 

|                 | ACC Top1 | ACC Top5 | Recall | AUC    |
| --------------- | -------- | -------- | ------ | ------ |
| DenseNet121     | 98.46%   | 99.96%   | 98.46% | 99.86% |
| DenseNet121+DRA | 18.03%   | 77.92%   | 18.03% | 73.67% |
| DenseNet121+PGD | 65.88%   | 94.07%   | 65.88% | 92.84% |

因为属于无目标攻击所以TOP 5下降并不多。




## Citation

If you find our work and this repository useful. Please consider giving a star ⭐ and citation.

> ```
> @article{zhu2022boosting,
>   title={Towards Understanding and Boosting Adversarial Transferability from a Distribution Perspective},
>   author={Yao Zhu, Yuefeng Chen, Xiaodan Li, Kejiang Chen, Yuan He, Xiang Tian, Bolun Zheng, Yaowu Chen, Qingming Huang},
>   booktitle={IEEE Transaction on Image Processing},
>   year={2022}
> }
> ```