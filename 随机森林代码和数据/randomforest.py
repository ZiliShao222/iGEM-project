import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# 第一步：数据加载与清洗
# ==========================================

def load_data_safely():
    """安全地加载数据文件（简化版：直接读取 Data.xlsx）"""
    file_path = 'Data.xlsx'
    if not os.path.exists(file_path):
        print(f"错误: 未找到文件 {file_path}")
        return None
    try:
        # 使用 header=None 以保证原始数据完整（第一行为列名）
        df = pd.read_excel(file_path, header=None)
        print(f"成功读取Excel文件，形状: {df.shape}")
        return df
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None

# 1. 读取数据（使用安全加载函数）
df_raw = load_data_safely()

if df_raw is None:
    print("无法读取数据，程序退出")
    exit()

# 显示原始数据信息
print(f"\n原始数据形状: {df_raw.shape}")

# 2. 转置数据 (Transpose)
df_T = df_raw.T

print(f"\n转置后数据形状: {df_T.shape}")

# 3. 设置列名
# 注意：第一行包含列名
df_T.columns = df_T.iloc[0]  # 取第一行作为列名
df_T = df_T.drop(df_T.index[0])  # 删除原标题行
df_T = df_T.reset_index(drop=True)

print(f"\n设置列名后数据形状: {df_T.shape}")
print("\n列名:")
print(df_T.columns.tolist())

# 4. 重命名日期列
# 如果第一列叫 'Date'，将其改名
if 'Date' in df_T.columns:
    df_T.rename(columns={'Date': 'Date_Label'}, inplace=True)

# 5. 剔除无效列
cols_to_drop = [
    'n-Alkanes and Fatty acids, μg/g', 
    'PHAs, μg/g', 
    'elements, μg/g',
    'Date_Label'
]

# 只删除存在的列
actual_cols_to_drop = [c for c in cols_to_drop if c in df_T.columns]
if actual_cols_to_drop:
    print(f"\n将要删除的列: {actual_cols_to_drop}")
    df_model = df_T.drop(columns=actual_cols_to_drop)
else:
    df_model = df_T.copy()

print(f"\n删除无效列后数据形状: {df_model.shape}")

# 6. 核心修复步骤：强制列名为字符串
df_model.columns = df_model.columns.astype(str)

# 7. 数据类型转换 (全部转为数字)
df_model = df_model.apply(pd.to_numeric, errors='coerce')

# 检查转换结果
print(f"\n数据类型转换后形状: {df_model.shape}")
print("\n各列数据类型:")
print(df_model.dtypes.head(20))  # 显示前20列的数据类型

# 8. 分离特征 (X) 和 目标 (y)
target_col = 'DTT consumption, nmolDTT/min*m3 of air'

# 检查目标列是否存在
if target_col not in df_model.columns:
    # 如果目标列不存在，尝试查找类似的列名
    print(f"\n错误: 目标列 '{target_col}' 不存在")
    print("可用的列名:")
    for col in df_model.columns:
        if 'DTT' in col or 'consumption' in col:
            print(f"  {col}")
    
    # 尝试使用第一个可能的目标列
    dtt_cols = [col for col in df_model.columns if 'DTT' in col or 'consumption' in col]
    if dtt_cols:
        target_col = dtt_cols[0]
        print(f"\n使用替代目标列: {target_col}")
    else:
        print("错误: 未找到目标列，程序退出")
        exit()

# 删除目标值为空的行
print(f"\n删除目标值为空的行前数据形状: {df_model.shape}")
initial_rows = df_model.shape[0]
df_model = df_model.dropna(subset=[target_col])
final_rows = df_model.shape[0]
print(f"删除目标值为空的行后数据形状: {df_model.shape}")
print(f"删除了 {initial_rows - final_rows} 行")

y = df_model[target_col]
X = df_model.drop(columns=[target_col])

# 填充特征中的缺失值为0
X = X.fillna(0)

print(f"\n最终数据形状:")
print(f"X (特征): {X.shape}")
print(f"y (目标): {y.shape}")

# 检查数据
print(f"\n目标变量统计:")
print(f"最小值: {y.min():.6f}")
print(f"最大值: {y.max():.6f}")
print(f"平均值: {y.mean():.6f}")
print(f"标准差: {y.std():.6f}")

print(f"\n特征矩阵统计:")
print(f"非零值比例: {(X != 0).sum().sum() / (X.shape[0] * X.shape[1]):.2%}")

# ==========================================
# 第二步：模型训练与评估
# ==========================================

print("\n" + "="*50)
print("开始模型训练")
print("="*50)

# --- Lasso 回归 (适合小样本) ---
print("\n训练 Lasso 回归...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.1, max_iter=10000, random_state=42) 
lasso.fit(X_scaled, y)
y_pred_lasso = lasso.predict(X_scaled)
r2_lasso = r2_score(y, y_pred_lasso)
mse_lasso = mean_squared_error(y, y_pred_lasso)
print(f"Lasso R² (拟合优度): {r2_lasso:.4f}")
print(f"Lasso MSE: {mse_lasso:.6f}")

# --- 随机森林 (Random Forest) ---
print("\n训练随机森林回归...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
y_pred_rf = rf.predict(X)
r2_rf = r2_score(y, y_pred_rf)
mse_rf = mean_squared_error(y, y_pred_rf)
print(f"Random Forest R²: {r2_rf:.4f}")
print(f"Random Forest MSE: {mse_rf:.6f}")

# ==========================================
# 第三步：特征重要性可视化
# ==========================================

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]  # 降序排列

# 取前 10 个最重要的特征
top_n = min(10, len(X.columns))
top_features = [X.columns[i] for i in indices[:top_n]]
top_importances = importances[indices[:top_n]]

# 创建图形
plt.figure(figsize=(12, 8))
plt.title("Top 10 Key Contributors to DTT Toxicity", fontsize=16, fontweight='bold')
bars = plt.barh(range(top_n), top_importances, align="center", color='skyblue', edgecolor='black')
plt.yticks(range(top_n), top_features, fontsize=12)
plt.xlabel("Relative Importance", fontsize=14)
plt.gca().invert_yaxis()  # 最重要的在顶部

# 在条形上添加数值
for i, (bar, importance) in enumerate(zip(bars, top_importances)):
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{importance:.4f}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.show()

# 打印前10个重要特征
print("\n随机森林特征重要性排名 (前10):")
print("-" * 60)
for i in range(top_n):
    print(f"{i+1:2d}. {top_features[i]:40s}: {top_importances[i]:.6f}")

# ==========================================
# 第四步：额外的诊断信息
# ==========================================

print("\n" + "="*50)
print("模型诊断信息")
print("="*50)

# 1. 检查特征相关性
print("\n1. 目标变量与Top 5特征的相关系数:")
for i in range(min(5, top_n)):
    feature = top_features[i]
    correlation = np.corrcoef(X[feature], y)[0, 1]
    print(f"   {feature:30s}: {correlation:.4f}")

# 2. 检查数据分布
print(f"\n2. 数据分布:")
print(f"   样本数量: {X.shape[0]}")
print(f"   特征数量: {X.shape[1]}")
print(f"   样本/特征比: {X.shape[0]/X.shape[1]:.2f}")

# 3. 检查是否有常数特征
constant_features = [col for col in X.columns if X[col].nunique() == 1]
if constant_features:
    print(f"\n3. 发现常数特征 (将被删除): {len(constant_features)} 个")
    # 在训练模型前应该删除这些特征
else:
    print(f"\n3. 没有常数特征")