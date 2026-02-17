# 库导入
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_mean_pool
import matplotlib.pyplot as plt

# 构建图结构
def build_graph_structure():
    """
    输出：
    - edge_index: 边连接索引（torch.long），形状[2, E]，E为边总数
    """
    # 1. 定义节点映射
    node_dic = {"Cu": 0, "Fe": 1, "1,2-NQN": 2, "1,4-NQN": 3, "PQN": 4, "•OH": 5}
    num_nodes = len(node_dic)  # 6个节点,5个活性物质+1个•OH

    # 2. 构建边索引
    edge_index = []
    # 第一部分：0-4（5个活性物质节点）两两全连接
    for i in range(5):
        for j in range(i + 1, 5):
            edge_index.append([i, j])  # 正向边
            edge_index.append([j, i])  # 反向边（无向图）
    # 第二部分：0-4分别与5（•OH节点）连接
    for i in range(5):
        edge_index.append([i, 5])  # 活性物质→•OH
    # 转换为PyG要求的tensor格式（[2, E]）
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return edge_index

# 初步设定节点特征为[M,5,1]

edge_index = build_graph_structure()
num_nodes = 6  # 固定6个节点（Cu/Fe/1,2-NQN/1,4-NQN/PQN/•OH）
nodes_names = ["Cu", "Fe", "1,2-NQN", "1,4-NQN", "PQN", "•OH"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
edge_index = edge_index.to(device)
dimention = 6 # 节点特征维度（原有3维 + PM2.5 + SOA_mass + SOA_activity = 6维）

print(f"边索引形状：{edge_index.shape}，节点数：{num_nodes}，设备：{device}")

# 数据准备与处理
# 生成完整数据集（1820条样本：1500单化合物 + 320混合物）
print("正在生成数据集...")

def generate_complete_dataset(sample_num_per_comp=300):
    """生成完整数据集（包含SOA特征）"""
    dataset = []
    np.random.seed(42)

    # SOA参数范围（基于论文Table S6）
    soa_mass_range = (0.1, 0.5)  # SOA质量分数 10%~50%
    soa_activity_range = (0.1, 0.4)  # SOA活性系数 0.1~0.4

    # ========== 单化合物数据 (1500条) ==========
    # 1. Cu(II)
    cu_conc = np.random.uniform(low=0.8, high=10000, size=sample_num_per_comp)
    for conc in cu_conc:
        pm25 = conc + np.random.uniform(50, 500)  # PM2.5浓度（Cu + 其他成分）
        soa_mass = np.random.uniform(*soa_mass_range)
        soa_activity = np.random.uniform(*soa_activity_range)
        base_oh = max(0.0, 0.0383 * np.log(conc) + 0.009)
        # SOA调整：稀释效应 + 活性促进
        dilution = 1 - (soa_mass / 100) * 0.5
        activity = 1 + soa_activity * 0.2
        oh = max(0.0, base_oh * dilution * activity)
        dataset.append([round(conc, 4), 0.0, 0.0, 0.0, 0.0, round(oh, 6),
                       round(pm25, 2), round(soa_mass, 3), round(soa_activity, 3)])

    # 2. Fe(II)
    fe_conc = np.random.uniform(low=0, high=10000, size=sample_num_per_comp)
    for conc in fe_conc:
        pm25 = conc + np.random.uniform(50, 500)
        soa_mass = np.random.uniform(*soa_mass_range)
        soa_activity = np.random.uniform(*soa_activity_range)
        base_oh = max(0.0, (-3.13 * 10**-8) * (conc ** 2) + (6.86 * 10**-4) * conc)
        dilution = 1 - (soa_mass / 100) * 0.5
        activity = 1 + soa_activity * 0.2
        oh = max(0.0, base_oh * dilution * activity)
        dataset.append([0.0, round(conc, 4), 0.0, 0.0, 0.0, round(oh, 6),
                       round(pm25, 2), round(soa_mass, 3), round(soa_activity, 3)])

    # 3. 1,2-NQN
    nqn12_conc = np.random.uniform(low=0, high=500, size=sample_num_per_comp)
    for conc in nqn12_conc:
        pm25 = conc + np.random.uniform(50, 500)
        soa_mass = np.random.uniform(*soa_mass_range)
        soa_activity = np.random.uniform(*soa_activity_range)
        base_oh = max(0.0, 0.486 * (1 - np.exp(-0.0191 * conc)))
        dilution = 1 - (soa_mass / 100) * 0.5
        activity = 1 + soa_activity * 0.2
        oh = max(0.0, base_oh * dilution * activity)
        dataset.append([0.0, 0.0, round(conc, 4), 0.0, 0.0, round(oh, 6),
                       round(pm25, 2), round(soa_mass, 3), round(soa_activity, 3)])

    # 4. 1,4-NQN
    nqn14_conc = np.random.uniform(low=0, high=500, size=sample_num_per_comp)
    for conc in nqn14_conc:
        pm25 = conc + np.random.uniform(50, 500)
        soa_mass = np.random.uniform(*soa_mass_range)
        soa_activity = np.random.uniform(*soa_activity_range)
        base_oh = max(0.0, (2.59 * 10**-4) * conc)
        dilution = 1 - (soa_mass / 100) * 0.5
        activity = 1 + soa_activity * 0.2
        oh = max(0.0, base_oh * dilution * activity)
        dataset.append([0.0, 0.0, 0.0, round(conc, 4), 0.0, round(oh, 6),
                       round(pm25, 2), round(soa_mass, 3), round(soa_activity, 3)])

    # 5. PQN
    pqn_conc = np.random.uniform(low=0, high=500, size=sample_num_per_comp)
    for conc in pqn_conc:
        pm25 = conc + np.random.uniform(50, 500)
        soa_mass = np.random.uniform(*soa_mass_range)
        soa_activity = np.random.uniform(*soa_activity_range)
        base_oh = max(0.0, 0.312 * (1 - np.exp(-0.0053 * conc)))
        dilution = 1 - (soa_mass / 100) * 0.5
        activity = 1 + soa_activity * 0.2
        oh = max(0.0, base_oh * dilution * activity)
        dataset.append([0.0, 0.0, 0.0, 0.0, round(conc, 4), round(oh, 6),
                       round(pm25, 2), round(soa_mass, 3), round(soa_activity, 3)])

    # ========== 混合物数据 (320条) ==========
    # Table S1 原始数据 (20条)
    mixture_data = [
        [100, 250, 0, 0, 0, 0.49],
        [300, 250, 0, 0, 0, 0.67],
        [74, 173, 0, 0, 0, 0.41],
        [79, 89, 0, 0, 0, 0.36],
        [242, 65, 0, 0, 0, 0.46],
        [17, 13, 0, 0, 0, 0.14],
        [74, 49, 0, 0, 0, 0.31],
        [500, 500, 0, 0, 0, 0.90],
        [1000, 1000, 0, 0, 0, 1.6],
        [2000, 1000, 0, 0, 0, 2.2],
        [2000, 2000, 0, 0, 0, 2.4],
        [39, 1022, 0, 3.7, 0.4, 0.56],
        [50, 0, 20, 0, 0, 0.39],
        [150, 0, 20, 0, 0, 0.48],
        [500, 0, 20, 0, 0, 0.70],
        [222, 549, 15, 0, 1.1, 1.2],
        [442, 459, 2.8, 3.4, 0.2, 1.1],
        [500, 500, 20, 0, 0, 1.4],
        [500, 0, 0, 500, 0, 1.2],
        [500, 500, 0, 500, 0, 1.7],
    ]

    # 为每个原始样本生成74个扰动样本（20原始+1480扰动=1500）
    for fe, cu, nqn12, nqn14, pqn, oh in mixture_data:
        pm25 = cu + fe + nqn12 + nqn14 + pqn + np.random.uniform(50, 500)
        soa_mass = np.random.uniform(*soa_mass_range)
        soa_activity = np.random.uniform(*soa_activity_range)
        # SOA调整后的•OH
        dilution = 1 - (soa_mass / 100) * 0.5
        activity = 1 + soa_activity * 0.2
        adjusted_oh = max(0.0, oh * dilution * activity)

        # 添加原始样本
        dataset.append([round(cu, 4), round(fe, 4), round(nqn12, 4),
                     round(nqn14, 4), round(pqn, 4), round(adjusted_oh, 6),
                     round(pm25, 2), round(soa_mass, 3), round(soa_activity, 3)])

        # 生成74个扰动样本（每个原始样本）
        for _ in range(74):
            noise_ratio = 0.05  # 扰动范围±5%
            new_cu = max(0, cu + cu * np.random.uniform(-noise_ratio, noise_ratio))
            new_fe = max(0, fe + fe * np.random.uniform(-noise_ratio, noise_ratio))
            new_nqn12 = max(0, nqn12 + nqn12 * np.random.uniform(-noise_ratio, noise_ratio))
            new_nqn14 = max(0, nqn14 + nqn14 * np.random.uniform(-noise_ratio, noise_ratio))
            new_pqn = max(0, pqn + pqn * np.random.uniform(-noise_ratio, noise_ratio))
            new_pm25 = new_cu + new_fe + new_nqn12 + new_nqn14 + new_pqn + np.random.uniform(50, 500)
            new_soa_mass = max(0.01, min(0.5, soa_mass + soa_mass * np.random.uniform(-0.1, 0.1)))
            new_soa_activity = max(0.1, min(0.4, soa_activity + soa_activity * np.random.uniform(-0.1, 0.1)))

            # SOA调整后的•OH
            new_dilution = 1 - (new_soa_mass / 100) * 0.5
            new_activity = 1 + new_soa_activity * 0.2
            new_oh = max(0.0, oh * new_dilution * new_activity)

            dataset.append([round(new_cu, 4), round(new_fe, 4), round(new_nqn12, 4),
                         round(new_nqn14, 4), round(new_pqn, 4), round(new_oh, 6),
                         round(new_pm25, 2), round(new_soa_mass, 3), round(new_soa_activity, 3)])

    # 转换为DataFrame并打乱
    df = pd.DataFrame(dataset, columns=["Cu", "Fe", "1,2-NQN", "1,4-NQN", "PQN", "•OH", "PM2.5", "SOA_mass", "SOA_activity"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# 生成完整数据集
df = generate_complete_dataset(sample_num_per_comp=300)

print(f"数据集大小：{df.shape}，数据集预览（前3行）：\n{df.head(3)}")
print(f"\n数据列：{df.columns.tolist()}")
print(f"特征维度：节点特征(5) + SOA全局特征(3) = 8维输入")
print("数据集加载完成")

# 保存数据集为CSV文件
df.to_csv('dataset.csv', index=False, encoding='utf-8-sig')
print(f"\n数据集已保存为 dataset.csv")
print(f"文件位置: {os.path.abspath('dataset.csv')}")

# ========== 检查原始混合物数据与单物质公式的关系 ==========
print("\n=== 检查原始混合物数据与单物质公式的关系 ===")
mixture_original = [
    [100, 250, 0, 0, 0, 0.49],
    [300, 250, 0, 0, 0, 0.67],
    [74, 173, 0, 0, 0, 0.41],
    [79, 89, 0, 0, 0, 0.36],
    [242, 65, 0, 0, 0, 0.46],
    [17, 13, 0, 0, 0, 0.14],
    [74, 49, 0, 0, 0, 0.31],
    [500, 500, 0, 0, 0, 0.90],
    [1000, 1000, 0, 0, 0, 1.6],
    [2000, 1000, 0, 0, 0, 2.2],
    [2000, 2000, 0, 0, 0, 2.4],
    [39, 1022, 0, 3.7, 0.4, 0.56],
    [50, 0, 20, 0, 0, 0.39],
    [150, 0, 20, 0, 0, 0.48],
    [500, 0, 20, 0, 0, 0.70],
    [222, 549, 15, 0, 1.1, 1.2],
    [442, 459, 2.8, 3.4, 0.2, 1.1],
    [500, 500, 20, 0, 0, 1.4],
    [500, 0, 0, 500, 0, 1.2],
    [500, 500, 0, 500, 0, 1.7],
]

def predict_oh_formula(fe, cu, nqn12, nqn14, pqn):
    """根据单物质公式计算•OH"""
    oh_cu = 0.0383 * np.log(cu) + 0.009 if cu > 0 else 0
    oh_fe = (-3.13 * 10**-8) * (fe ** 2) + (6.86 * 10**-4) * fe if fe > 0 else 0
    oh_nqn12 = 0.486 * (1 - np.exp(-0.0191 * nqn12)) if nqn12 > 0 else 0
    oh_nqn14 = (2.59 * 10**-4) * nqn14 if nqn14 > 0 else 0
    oh_pqn = 0.312 * (1 - np.exp(-0.0053 * pqn)) if pqn > 0 else 0
    return oh_cu + oh_fe + oh_nqn12 + oh_nqn14 + oh_pqn

print("对比实际•OH与公式计算•OH（假设加和）：")
print(f"{'Cu':>6} {'Fe':>6} {'N12':>6} {'N14':>6} {'PQN':>6} {'实际OH':>8} {'公式OH':>8} {'差异':>8}")
print("-" * 60)
diffs = []
for fe, cu, nqn12, nqn14, pqn, oh_actual in mixture_original:
    oh_formula = predict_oh_formula(fe, cu, nqn12, nqn14, pqn)
    diff = oh_actual - oh_formula
    diffs.append(abs(diff))
    print(f"{cu:6.0f} {fe:6.0f} {nqn12:6.1f} {nqn14:6.1f} {pqn:6.1f} {oh_actual:8.2f} {oh_formula:8.2f} {diff:8.2f}")

print(f"\n平均绝对误差: {np.mean(diffs):.4f}")
print(f"最大绝对误差: {np.max(diffs):.4f}")
print(f"最小绝对误差: {np.min(diffs):.4f}")
print(f"误差标准差: {np.std(diffs):.4f}")

# 构建GAT模型
from torch_geometric.nn import GATConv as PyGGATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, heads=4, dropout=0.1):
        super().__init__()
        # 使用PyG内置的GAT卷积层
        self.conv1 = PyGGATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=False)
        self.conv2 = PyGGATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout, concat=False)
        self.conv3 = PyGGATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout, concat=False)
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch=None)
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x
    
model = GAT(in_channels=dimention, hidden_channels=64, out_channels=1, heads=4, dropout=0.1).to(device)  # dimention=6
print(model)
print("GAT模型构建完成")

# 测试模型前向传播
print("\n=== 测试模型前向传播 ===")
x = torch.randn(num_nodes, dimention).to(device)  # 随机生成节点特征 [6, 3]
output = model(x, edge_index)
print(f"输入形状: {x.shape}, 输出形状: {output.shape}")
print("模型测试通过")

# 添加损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
print("\n损失函数和优化器配置完成")

# 训练和评估函数
def train(x, y):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(x, y):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        loss = criterion(out, y)
    return loss.item()

print("\n=== 基础框架搭建完成 ===")
print("提示：加载数据集后可以开始训练")

# 数据预处理和转换为图输入格式（加入SOA特征）
def prepare_data_for_gnn(df):
    """
    将DataFrame转换为GNN输入格式（包含SOA全局特征）
    节点特征扩展为6维：
    - 第1维：标准化浓度
    - 第2维：是否与Fe共存（0/1）
    - 第3维：物种类型嵌入（1=Cu, 2=Fe, 3=醌类）
    - 第4-6维：全局特征（PM2.5, SOA_mass, SOA_activity）
    :param df: 包含[Cu, Fe, 1,2-NQN, 1,4-NQN, PQN, •OH, PM2.5, SOA_mass, SOA_activity]的DataFrame
    :return: x_train, y_train, x_val, y_val, scaler_feature, scaler_global
    """
    # 分离特征和标签
    feature_cols = ["Cu", "Fe", "1,2-NQN", "1,4-NQN", "PQN"]
    global_feature_cols = ["PM2.5", "SOA_mass", "SOA_activity"]
    target_col = "•OH"

    X = df[feature_cols].values
    X_global = df[global_feature_cols].values
    y = df[target_col].values.reshape(-1, 1)

    # 定义物种类型嵌入（Cu=1, Fe=2, 醌类=3）
    species_embedding = [1, 2, 3, 3, 3]

    # 先划分训练集/验证集（避免数据泄露）
    all_idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(all_idx, test_size=0.2, random_state=42)

    # 分离训练集和验证集
    X_train = X[train_idx]
    X_global_train = X_global[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    X_global_val = X_global[val_idx]
    y_val = y[val_idx]

    print(f"训练集大小: {len(train_idx)}, 验证集大小: {len(val_idx)}")

    # 限定有效浓度范围（按论文Table 1）
    X_train = np.clip(X_train, 0, [10000, 10000, 500, 500, 500])
    X_val = np.clip(X_val, 0, [10000, 10000, 500, 500, 500])

    # 分别标准化节点特征和全局特征
    scaler_feature = StandardScaler()
    X_train_scaled = scaler_feature.fit_transform(X_train)
    X_val_scaled = scaler_feature.transform(X_val)

    scaler_global = StandardScaler()
    X_global_train_scaled = scaler_global.fit_transform(X_global_train)
    X_global_val_scaled = scaler_global.transform(X_global_val)

    # 构建训练集节点特征（6维：原有3维 + 全局特征3维）
    x_train = []
    for i in range(len(X_train_scaled)):
        # 原有3维特征
        x_sample = np.zeros((num_nodes, 3))
        x_sample[:5, 0] = X_train_scaled[i, :]
        has_fe = 1.0 if X_train[i, 1] > 0 else 0.0
        x_sample[:5, 1] = has_fe
        x_sample[:5, 2] = species_embedding
        x_sample[5, :] = 0  # •OH节点特征为0

        # 拼接全局特征（广播到所有节点）
        x_sample = np.hstack([x_sample, np.tile(X_global_train_scaled[i], (num_nodes, 1))])

        x_train.append(torch.tensor(x_sample, dtype=torch.float32).to(device))

    # 构建验证集节点特征
    x_val = []
    for i in range(len(X_val_scaled)):
        x_sample = np.zeros((num_nodes, 3))
        x_sample[:5, 0] = X_val_scaled[i, :]
        has_fe = 1.0 if X_val[i, 1] > 0 else 0.0
        x_sample[:5, 1] = has_fe
        x_sample[:5, 2] = species_embedding
        x_sample[5, :] = 0
        x_sample = np.hstack([x_sample, np.tile(X_global_val_scaled[i], (num_nodes, 1))])

        x_val.append(torch.tensor(x_sample, dtype=torch.float32).to(device))

    # 转换标签为tensor
    y_train = [torch.tensor(y_train[i:i+1], dtype=torch.float32).to(device) for i in range(len(y_train))]
    y_val = [torch.tensor(y_val[i:i+1], dtype=torch.float32).to(device) for i in range(len(y_val))]

    # 统计单化合物/混合物样本数
    single_mask = ((df[feature_cols] > 0).sum(axis=1) == 1).values
    train_single = len([i for i in train_idx if single_mask[i]])
    train_mixture = len(train_idx) - train_single
    val_single = len([i for i in val_idx if single_mask[i]])
    val_mixture = len(val_idx) - val_single

    print(f"训练集: 单化合物={train_single}, 混合物={train_mixture}")
    print(f"验证集: 单化合物={val_single}, 混合物={val_mixture}")

    # 验证第一个样本特征
    print("\n=== 验证节点特征维度和数值 ===")
    print(f"训练集第一个样本节点特征形状: {x_train[0].shape}")
    print("节点特征矩阵（前2行示例）:")
    print(x_train[0].cpu().numpy()[:2, :])
    print("特征含义：[标准化浓度, 是否与Fe共存, 物种类型, PM2.5, SOA_mass, SOA_activity]")

    return x_train, y_train, x_val, y_val, scaler_feature, scaler_global

# 准备训练数据
x_train, y_train, x_val, y_val, scaler_feature, scaler_global = prepare_data_for_gnn(df)

# 训练循环
def train_model(x_train, y_train, x_val, y_val, epochs=200, early_stopping_patience=30):
    """
    训练模型
    参数:
        x_train: 训练集节点特征
        y_train: 训练集目标值
        x_val: 验证集节点特征
        y_val: 验证集目标值
        epochs: 训练轮数
        early_stopping_patience: 早停阈值
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # 训练
        total_train_loss = 0
        for i in range(len(x_train)):
            loss = train(x_train[i], y_train[i])
            total_train_loss += loss
        avg_train_loss = total_train_loss / len(x_train)
        train_losses.append(avg_train_loss)

        # 验证
        total_val_loss = 0
        for i in range(len(x_val)):
            loss = evaluate(x_val[i], y_val[i])
            total_val_loss += loss
        avg_val_loss = total_val_loss / len(x_val)
        val_losses.append(avg_val_loss)

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # 打印进度（每10个epoch）
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # 早停
        if patience_counter >= early_stopping_patience:
            print(f"早停触发于epoch {epoch+1}")
            break

    # 训练完成后绘制loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    print("训练曲线已保存为 training_history.png")

    return best_val_loss, train_losses, val_losses

# 预测函数
def predict(x):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    return out

# 可视化训练过程
def plot_training_history(train_losses, val_losses):
    # 图一，训练和验证损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    print("训练曲线已保存为 training_history.png")

# 预测结果可视化
def plot_predictions(y_true, y_pred, title="Prediction Results"):

    # 图二，真实值对比预测值
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true.numpy(), y_pred.numpy(), alpha=0.6, edgecolors='k', linewidth=0.5)

    # 添加对角线
    min_val = min(y_true.min().item(), y_pred.min().item())
    max_val = max(y_true.max().item(), y_pred.max().item())
    plt.plot([min_val, max_val], [min_val, max_val], '--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)

    # 计算并显示指标（修复R²计算，添加eps避免除以零）
    mse = torch.mean((y_true - y_pred) ** 2).item()
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2) + 1e-8  # 添加1e-8避免除以零
    r2 = 1 - (ss_res / ss_tot)

    textstr = f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()
    print("预测结果可视化已保存为 predictions.png")

# 开始训练
if __name__ == "__main__":
    print("\n=== 开始训练 ===")

    # 训练模型
    best_val_loss, train_losses, val_losses = train_model(
        x_train, y_train, x_val, y_val,
        epochs=200, early_stopping_patience=30
    )

    print(f"\n最佳验证损失: {best_val_loss:.4f}")

    # 评估模型
    print("\n=== 验证集评估 ===")

    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for i in range(len(x_val)):
            pred = predict(x_val[i])
            predictions.append(pred)
            targets.append(y_val[i])

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    # 计算指标
    mse = torch.mean((targets - predictions) ** 2).item()
    rmse = torch.sqrt(torch.mean((targets - predictions) ** 2)).item()
    mae = torch.mean(torch.abs(targets - predictions)).item()
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2) + 1e-8
    r2 = 1 - (ss_res / ss_tot)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    # 可视化预测结果（训练曲线已在训练过程中实时显示）
    plot_predictions(targets, predictions, title="Validation Predictions")

    # 保存训练好的模型和scaler
    torch.save(model.state_dict(), 'best_gat_model.pth')
    print(f"\n模型已保存为 best_gat_model.pth")

    import pickle
    with open('scaler_feature.pkl', 'wb') as f:
        pickle.dump(scaler_feature, f)
    print(f"特征Scaler已保存为 scaler_feature.pkl")

    with open('scaler_global.pkl', 'wb') as f:
        pickle.dump(scaler_global, f)
    print(f"全局特征Scaler已保存为 scaler_global.pkl")
    print("\n训练完成！现在可以运行 igemgraph_predict.py 进行预测")