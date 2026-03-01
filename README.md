# iGEM Project - 基于机器学习的预测模型

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 📖 项目简介

本项目是 iGEM（国际遗传工程机器大赛）的机器学习预测模型项目，主要探索化学物质对环境因子的影响预测。项目包含两种主要的建模方法：

- **随机森林模型**：基于传统机器学习的预测方法
- **图神经网络（GAT）模型**：基于图结构的深度学习方法

## Zili Shao 工作
1. 寻找背景论文，支撑dry lab模型的生物学基础
2. 根据Pershen LI寻找的原始论文数据进行处理生成初步数据集
3. 寻找论文补充更多维数据
4. GAT
5. Linear Regression
6. 海报制作

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
    ├── igemOHSOR.py           # OH SOR 关联分析
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
- MSE: 0.0134
- R²: 0.9847
- RMSE: 0.1157

### GAT 模型
- 最优损失函数: 0.102

## 📈 可视化结果

项目包含以下可视化结果：

- `OHSOR.png` - OH SOR 结果分析
- `ROStoCell.png` - ROS 细胞转化分析
- `secpollution.png` - 二次污染分析

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

### 背景支撑论文

1. **Incomplete solid fuel burning as the major emission source of atmospheric phenols and environmentally persistent free radicals**
   - 结论：木质素不完全燃烧生成酚类前体物导致EPFRs产生

2. **Molecular Products and Radicals from Pyrolysis of Lignin**
   - 结论：居民固体燃料（生物质、煤）不完全燃烧是大气EPFRs（80.6%）和酚类（99.9%）的主要来源

3. **Oxidants, oxidative stress and the biology of ageing**
   - 结论：首次系统阐述ROS作为氧化剂，通过氧化损伤蛋白质、脂质膜和DNA引发细胞衰老，明确其与代谢疾病、衰老相关病理过程的关联

4. **Oxidative stress**
   - 结论：全面解析氧化应激的生物学机制，指出过量ROS会打破机体抗氧化平衡，诱发慢性炎症、细胞凋亡

5. **Aerosol health effects from molecular to global scales**
   - 结论：聚焦大气气溶胶健康效应，明确ROS是PM₂.₅引发呼吸系统损伤、心血管疾病的关键中间物质

6. **Chemical exposure-response relationship between air pollutants and reactive oxygen species in the human respiratory tract**
   - 结论：开发KM-SUB-ELF模型，模拟显示肺部ROS浓度与呼吸道疾病、心血管疾病直接相关

7. **Environmentally Persistent Free Radicals, Reactive Oxygen Species Generation, and Oxidative Potential of Highway PM2.5**
   - 结论：EPFRs可能与柴油尾气颗粒相关，并通过与Fe、Cu等过渡金属相互作用得以稳定

8. **Hydroxyl radical generations form the physiologically relevant Fenton-like reactions**
   - 结论：·OH毒性机制

9. **Hydroxyl radical is a significant player in oxidative DNA damage in vivo**
   - 结论：·OH与DNA损伤

10. **Effects of the Cytoplasm and Mitochondrial Specific Hydroxyl Radical Scavengers... in Bleomycin-Induced Pulmonary Fibrosis Model Mice**
    - 结论：·OH与肺部疾病

### 技术参考

- Graph Attention Networks (GAT): Velickovic et al., 2018
- iGEM Official Website: https://igem.org

## 👥 团队成员

### Dry Lab
- **邵子栗** - GAT 模型开发
- **李沛珅** - 随机森林模型开发

### Wet Lab
- **梁榆闰**
- **陈志衡**

### Human Practice
- **杨橙子**
- **杨戈文卓**

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
