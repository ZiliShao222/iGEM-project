# GAT模型预测程序（加载已训练模型进行预测）
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv as PyGGATConv, global_mean_pool
from torch.nn import Linear

# ====================== GAT模型定义（与训练时一致）======================
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, heads=4, dropout=0.1):
        super().__init__()
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

# ====================== 图结构和参数（与训练时一致）======================
def build_graph_structure():
    """构建图结构"""
    edge_index = []
    for i in range(5):
        for j in range(i + 1, 5):
            edge_index.append([i, j])
            edge_index.append([j, i])
    for i in range(5):
        edge_index.append([i, 5])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 固定参数
num_nodes = 6
dimention = 6  # 原有3维 + 全局特征3维 = 6维
edge_index = build_graph_structure()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
edge_index = edge_index.to(device)

# ====================== 加载模型和scaler======================
# 加载保存的scaler
import pickle
try:
    with open('scaler_feature.pkl', 'rb') as f:
        scaler_feature = pickle.load(f)
    print("✓ 特征Scaler加载成功")
except FileNotFoundError:
    print("✗ 错误：未找到scaler_feature.pkl，请先运行训练程序")
    exit(1)

try:
    with open('scaler_global.pkl', 'rb') as f:
        scaler_global = pickle.load(f)
    print("✓ 全局特征Scaler加载成功")
except FileNotFoundError:
    print("✗ 错误：未找到scaler_global.pkl，请先运行训练程序")
    exit(1)

# 加载模型
model = GAT(in_channels=dimention, hidden_channels=64, out_channels=1, heads=4, dropout=0.1).to(device)
try:
    model.load_state_dict(torch.load('best_gat_model.pth', map_location=device))
    model.eval()
    print("✓ 模型加载成功")
except FileNotFoundError:
    print("✗ 错误：未找到best_gat_model.pth，请先运行训练程序")
    exit(1)

# ====================== 预测函数 ======================
def predict_oh(cu, fe, nqn12, nqn14, pqn, pm25, soa_mass, soa_activity):
    """
    预测•OH浓度（包含SOA特征）
    参数:
        cu: Cu浓度 (mg/L)
        fe: Fe浓度 (mg/L)
        nqn12: 1,2-NQN浓度 (mg/L)
        nqn14: 1,4-NQN浓度 (mg/L)
        pqn: PQN浓度 (mg/L)
        pm25: PM2.5浓度 (μg/m³)
        soa_mass: SOA质量分数 (0~1)
        soa_activity: SOA活性系数 (0~1)
    返回:
        预测的•OH浓度 (mg/L)
    """
    # 准备节点特征
    X = np.array([cu, fe, nqn12, nqn14, pqn]).reshape(1, -1)
    X_scaled = scaler_feature.transform(X)

    # 准备全局特征
    X_global = np.array([pm25, soa_mass, soa_activity]).reshape(1, -1)
    X_global_scaled = scaler_global.transform(X_global)

    species_embedding = [1, 2, 3, 3, 3]

    # 构建节点特征（原有3维 + 全局特征3维）
    x_sample = np.zeros((num_nodes, 3))
    x_sample[:5, 0] = X_scaled[0, :]
    has_fe = 1.0 if X[0, 1] > 0 else 0.0
    x_sample[:5, 1] = has_fe
    x_sample[:5, 2] = species_embedding
    x_sample[5, :] = 0

    # 拼接全局特征
    x_sample = np.hstack([x_sample, np.tile(X_global_scaled[0], (num_nodes, 1))])

    x = torch.tensor(x_sample, dtype=torch.float32).to(device)

    # 预测
    with torch.no_grad():
        prediction = model(x, edge_index)

    return prediction.item()


def predict_cell_from_composition(cu, fe, nqn12, nqn14, pqn, pm25, soa_mass, soa_activity):
    """
    从物质组成预测细胞活性
    1. 预测•OH浓度
    2. 将•OH归一化为ROS浓度 (μmol/L)
    3. 使用ROS-细胞活性回归模型预测细胞活性

    参数:
        cu: Cu浓度 (mg/L)
        fe: Fe浓度 (mg/L)
        nqn12: 1,2-NQN浓度 (mg/L)
        nqn14: 1,4-NQN浓度 (mg/L)
        pqn: PQN浓度 (mg/L)
        pm25: PM2.5浓度 (μg/m³)
        soa_mass: SOA质量分数 (0~1)
        soa_activity: SOA活性系数 (0~1)

    返回:
        (•OH浓度 mg/L, ROS浓度 μmol/L, 细胞活性 %)
    """
    import pickle

    # 1. 预测•OH浓度
    oh_conc = predict_oh(cu, fe, nqn12, nqn14, pqn, pm25, soa_mass, soa_activity)

    # 2. 将•OH归一化为ROS浓度 (μmol/L)
    # 假设：•OH浓度 ≈ ROS浓度，需要进行单位转换
    # •OH分子量 = 17 g/mol, 1 mg/L = 1/17 mmol/L = (1/17)*1000 = 58.8 μmol/L
    oh_mw = 17  # •OH分子量 (g/mol)
    ros_concentration_umol = oh_conc / oh_mw * 1000  # mg/L → μmol/L

    # 3. 使用ROS-细胞活性回归模型预测细胞活性
    try:
        with open('ros_cell_model.pkl', 'rb') as f:
            ros_model = pickle.load(f)
        cell_viability = ros_model.predict([[ros_concentration_umol]])[0]
        # 确保细胞活性在合理范围内
        cell_viability = max(0, min(100, cell_viability))
    except FileNotFoundError:
        print("警告：未找到 ros_cell_model.pkl，请先运行 igemROStoCell.py 训练回归模型")
        cell_viability = None

    return oh_conc, ros_concentration_umol, cell_viability


def predict_sor_from_composition(cu, fe, nqn12, nqn14, pqn, pm25, soa_mass, soa_activity):
    """
    从物质组成预测硫氧化率（SOR）
    1. 预测•OH浓度
    2. 将•OH转换为•OH日间生产速率 (ppbV/h)
    3. 使用OH-SOR回归模型预测SOR

    参数:
        cu: Cu浓度 (mg/L)
        fe: Fe浓度 (mg/L)
        nqn12: 1,2-NQN浓度 (mg/L)
        nqn14: 1,4-NQN浓度 (mg/L)
        pqn: PQN浓度 (mg/L)
        pm25: PM2.5浓度 (μg/m³)
        soa_mass: SOA质量分数 (0~1)
        soa_activity: SOA活性系数 (0~1)

    返回:
        (•OH浓度 mg/L, •OH生产速率 ppbV/h, SOR %)
    """
    import pickle

    # 1. 预测•OH浓度
    oh_conc = predict_oh(cu, fe, nqn12, nqn14, pqn, pm25, soa_mass, soa_activity)

    # 2. 将•OH浓度转换为•OH日间生产速率 (ppbV/h)
    # 假设：•OH浓度与生产速率成正比
    # 根据论文数据，•OH浓度 ~1 mg/L 对应生产速率 ~0.7 ppbV/h
    # 转换系数：ppbV/h = mg/L * 0.7
    oh_production_rate = oh_conc * 0.7  # mg/L → ppbV/h

    # 3. 使用OH-SOR回归模型预测SOR
    try:
        with open('oh_sor_model.pkl', 'rb') as f:
            oh_sor_model = pickle.load(f)
        sor = oh_sor_model.predict([[oh_production_rate]])[0]
        # 确保SOR在合理范围内
        sor = max(0, min(100, sor))
    except FileNotFoundError:
        print("警告：未找到 oh_sor_model.pkl，请先运行 igemOHSOR.py 训练模型")
        sor = None

    return oh_conc, oh_production_rate, sor


def predict_all_outputs(cu, fe, nqn12, nqn14, pqn, pm25, soa_mass, soa_activity):
    """
    从物质组成预测所有输出：•OH、ROS、细胞活性、SOR、烯烃损失率

    参数:
        cu: Cu浓度 (mg/L)
        fe: Fe浓度 (mg/L)
        nqn12: 1,2-NQN浓度 (mg/L)
        nqn14: 1,4-NQN浓度 (mg/L)
        pqn: PQN浓度 (mg/L)
        pm25: PM2.5浓度 (μg/m³)
        soa_mass: SOA质量分数 (0~1)
        soa_activity: SOA活性系数 (0~1)

    返回:
        {
            'oh_conc': •OH浓度 (mg/L),
            'oh_conc_molecules': •OH浓度 (×10⁶ molecules/cm³),
            'ros_conc': ROS浓度 (μmol/L),
            'cell_viability': 细胞活性 (%),
            'oh_production_rate': •OH生产速率 (ppbV/h),
            'sor': 硫氧化率 (%),
            'alkene_loss_rate': 烯烃光化学损失率 (%)
        }
    """
    # 预测•OH
    oh_conc = predict_oh(cu, fe, nqn12, nqn14, pqn, pm25, soa_mass, soa_activity)

    # 转换为ROS浓度
    oh_mw = 17
    ros_conc = oh_conc / oh_mw * 1000  # mg/L → μmol/L

    # 转换为•OH生产速率
    oh_production_rate = oh_conc * 0.7  # mg/L → ppbV/h

    # 转换为•OH分子浓度 (×10⁶ molecules/cm³)
    # 1 mg/L = 1e-6 g/cm³, 分子量 = 17 g/mol
    # 分子数 = (1e-6 / 17) * 6.022e23 = 3.54e16 molecules/cm³
    # 转换为 ×10⁶ 单位：3.54e16 / 1e6 = 35400
    oh_conc_molecules = oh_conc * 35400 / 1e6  # mg/L → ×10⁶ molecules/cm³

    # 预测细胞活性
    try:
        with open('ros_cell_model.pkl', 'rb') as f:
            ros_model = pickle.load(f)
        cell_viability = ros_model.predict([[ros_conc]])[0]
        cell_viability = max(0, min(100, cell_viability))
    except FileNotFoundError:
        cell_viability = None

    # 预测SOR
    try:
        with open('oh_sor_model.pkl', 'rb') as f:
            oh_sor_model = pickle.load(f)
        sor = oh_sor_model.predict([[oh_production_rate]])[0]
        sor = max(0, min(100, sor))
    except FileNotFoundError:
        sor = None

    # 预测烯烃光化学损失率
    try:
        with open('oh_alkene_loss_model.pkl', 'rb') as f:
            alkene_model = pickle.load(f)
        alkene_loss_rate = alkene_model.predict([[oh_conc_molecules]])[0]
        alkene_loss_rate = max(0, min(100, alkene_loss_rate))
    except FileNotFoundError:
        alkene_loss_rate = None

    return {
        'oh_conc': oh_conc,
        'oh_conc_molecules': oh_conc_molecules,
        'ros_conc': ros_conc,
        'cell_viability': cell_viability,
        'oh_production_rate': oh_production_rate,
        'sor': sor,
        'alkene_loss_rate': alkene_loss_rate
    }

# ====================== 交互式预测 ======================
def main():
    print("\n" + "="*50)
    print("GAT模型预测完整输出（包含SOA特征）")
    print("="*50)
    print("\n请输入各物质浓度（mg/L）和SOA参数：")

    try:
        cu = float(input("Cu浓度 (mg/L): "))
        fe = float(input("Fe浓度 (mg/L): "))
        nqn12 = float(input("1,2-NQN浓度 (mg/L): "))
        nqn14 = float(input("1,4-NQN浓度 (mg/L): "))
        pqn = float(input("PQN浓度 (mg/L): "))
        pm25 = float(input("PM2.5浓度 (μg/m³): "))
        soa_mass = float(input("SOA质量分数 (0~1, 如0.35表示35%): "))
        soa_activity = float(input("SOA活性系数 (0~1, 推荐0.1~0.4): "))

        # 预测所有输出
        results = predict_all_outputs(cu, fe, nqn12, nqn14, pqn, pm25, soa_mass, soa_activity)

        print(f"\n{'='*50}")
        print(f"预测结果：")
        print(f"{'='*50}")
        print(f"•OH浓度            = {results['oh_conc']:.4f} mg/L")
        print(f"•OH浓度            = {results['oh_conc_molecules']:.2f} ×10⁶ molecules/cm³")
        print(f"ROS浓度            = {results['ros_conc']:.2f} μmol/L")
        if results['cell_viability'] is not None:
            print(f"细胞活性            = {results['cell_viability']:.2f}%")
        else:
            print(f"细胞活性            = 无法预测（请先运行 igemROStoCell.py）")
        print(f"•OH日间生产速率    = {results['oh_production_rate']:.4f} ppbV/h")
        if results['sor'] is not None:
            print(f"硫氧化率(SOR)       = {results['sor']:.2f}%")
        else:
            print(f"硫氧化率(SOR)       = 无法预测（请先运行 igemOHSOR.py）")
        if results['alkene_loss_rate'] is not None:
            print(f"烯烃光化学损失率    = {results['alkene_loss_rate']:.2f}%")
        else:
            print(f"烯烃光化学损失率    = 无法预测（请先运行 igemOHsecpollution.py）")
        print("="*50)

        # 返回预测值（可选：用于程序化调用）
        return results

    except ValueError:
        print("✗ 错误：请输入有效的数字")
        return None

if __name__ == "__main__":
    main()
