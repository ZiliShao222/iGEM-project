# iGEM Project - 基于机器学习的预测模型

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 📖 项目简介

本项目是 iGEM（国际遗传工程机器大赛）的机器学习预测模型项目，主要探索化学物质对环境因子的影响预测。项目包含两种主要的建模方法：

- **随机森林模型**：基于传统机器学习的预测方法
- **图神经网络（GAT）模型**：基于图结构的深度学习方法

## 🎯 研究目标

通过机器学习方法预测：
- OH 自由基对烯烃的损失率
- OH 污染物的产生
- ROS（活性氧）到细胞的转化

## 📁 项目结构

```
iGEM-project/
├── 随机森林代码和数据/
│   ├── Data.xlsx              # 原始数据集
│   └── randomforest.py        # 随机森林训练与预测脚本
└── GAT和线性回归层/
    ├── dataset.csv            # 图神经网络数据集
    ├── igemgraph.py           # GAT 模型训练
    ├── igemgraph_predict.py   # GAT 模型预测
    ├── igemOHsecpollution.py  # OH 二次污染分析
    ├── igemOHSOR.py           # OH SOR 预测
    ├── igemROStoCell.py       # ROS 细胞转化分析
    ├── best_gat_model.pth     # 训练好的 GAT 模型
    ├── best_gcn_model.pth     # 训练好的 GCN 模型
    ├── *.pkl                  # 各任务的训练模型
    └── *.png                  # 结果可视化图表
```

## 🔬 技术方案

### 1. 随机森林模型
- **算法**：Random Forest Regressor + Lasso
- **特点**：
  - 非线性建模能力强
  - 特征重要性可解释
  - 对异常值鲁棒
- **文件**：`随机森林代码和数据/randomforest.py`

### 2. 图神经网络（GAT）模型
- **算法**：Graph Attention Network
- **图结构**：
  - 6 个节点：Cu, Fe, 1,2-NQN, 1,4-NQN, PQN, •OH
  - 节点间全连接 + 与 •OH 的连接
- **特征维度**：6 维（化学特征 + 环境因子）
- **特点**：
  - 能够捕捉节点间的复杂关系
  - 注意力机制增强模型表达能力
- **文件**：`GAT和线性回归层/igemgraph.py`

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 1.10
torch-geometric
scikit-learn
pandas
numpy
matplotlib
openpyxl
```

### 安装依赖

```bash
pip install torch torchvision
pip install torch-geometric
pip install scikit-learn pandas numpy matplotlib openpyxl tqdm
```

### 运行随机森林模型

```bash
cd 随机森林代码和数据
python randomforest.py
```

### 运行 GAT 模型训练

```bash
cd GAT和线性回归层
python igemgraph.py
```

### 运行 GAT 模型预测

```bash
cd GAT和线性回归层
python igemgraph_predict.py
```

## 📊 模型性能

### 随机森林模型
- MSE: [待更新]
- R²: [待更新]

### GAT 模型
- 训练损失: [待更新]
- 验证损失: [待更新]

## 📈 可视化结果

项目包含以下可视化结果：

- `OHSOR.png` - OH SOR 预测结果
- `ROStoCell.png` - ROS 细胞转化分析
- `secpollution.png` - 二次污染预测

## 🔧 使用已训练模型

### 加载随机森林模型

```python
import pickle

with open('oh_alkene_loss_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 进行预测
# ...
```

### 加载 GAT 模型

```python
import torch

model = GAT(in_channels=6, hidden_channels=64)
model.load_state_dict(torch.load('best_gat_model.pth'))
model.eval()

# 进行预测
# ...
```

## 🧪 数据说明

### 数据来源
- 原始数据包含化学物质浓度、环境因子等特征
- 数据经过预处理和标准化

### 特征说明
- 节点特征：化学物质属性
- 全局特征：PM2.5、SOA_mass、SOA_activity

## 📝 参考文献

- Graph Attention Networks (GAT): Velickovic et al., 2018
- iGEM Official Website: https://igem.org

## 👥 团队成员

iGEM 团队成员

## 📄 License

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请通过以下方式联系：
- GitHub Issues: https://github.com/ZiliShao222/iGEM-project/issues

---

<div align="center">
  <i>Exploring AI Applications in iGEM Competition</i>
</div>
