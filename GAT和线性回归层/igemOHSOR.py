import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import pickle

# -------------------------- 1. 输入论文实测数据（来源：附录Table S7 + Table S4） --------------------------
# 数据说明：3组污染等级的OH日间生产速率（P_OH）和硫氧化率（SOR）均为论文直接给出的日均实测值
data = {
    'Pollution Level': ['SP', 'MP', 'HP'],
    'OH Production Rate (ppbV/h)': [0.40, 0.53, 1.09],  # 自变量：Table S7中Total生产速率均值
    'Sulfur Oxidation Rate (SOR, %)': [0.71, 0.79, 0.84]  # 因变量：Table S4中SOR均值
}
df = pd.DataFrame(data)

# 提取自变量（X）和因变量（y）
X = df['OH Production Rate (ppbV/h)'].values.reshape(-1, 1)  # 自变量需为二维数组（n_samples, n_features）
y = df['Sulfur Oxidation Rate (SOR, %)'].values  # 因变量为一维数组

# -------------------------- 2. 模型拟合（线性回归） --------------------------
# 初始化线性回归模型
model = LinearRegression()
# 拟合数据（基于论文实测数据）
model.fit(X, y)

# 提取模型参数
intercept = model.intercept_  # 截距（a）
slope = model.coef_[0]        # 斜率（b）

# 输出模型公式（贴合论文物理意义）
print("="*60)
print("Model 1: OH Production Rate -> Sulfur Oxidation Rate (SOR)")
print(f"Formula: SOR (%) = {intercept:.2f} + {slope:.2f} × P_OH (ppbV/h)")
print(f"Intercept a = {intercept:.4f}")
print(f"Slope b = {slope:.4f}")
print("="*60)

# -------------------------- 3. 模型验证（基于论文数据） --------------------------
# 用拟合模型预测SOR值
y_pred = model.predict(X)

# 计算模型性能指标（验证拟合效果）
r2 = r2_score(y, y_pred)                # 决定系数（越接近1，拟合越好）
mae = mean_absolute_error(y, y_pred)     # 平均绝对误差（越小，误差越小）

# 输出验证结果
print("\n[Model Validation] (Based on observed data)")
print(f"R² = {r2:.4f} (R²>0.7, good fit)")
print(f"MAE = {mae:.4f} (small error, accurate prediction)")

# 输出实测值与预测值对比表
df['SOR Pred (%)'] = y_pred.round(4)
df['Abs Error (%)'] = np.abs(df['Sulfur Oxidation Rate (SOR, %)'] - df['SOR Pred (%)']).round(4)
print("\n[Observed vs Predicted Values]")
print(df[['Pollution Level', 'OH Production Rate (ppbV/h)', 'Sulfur Oxidation Rate (SOR, %)', 'SOR Pred (%)', 'Abs Error (%)']])
print("="*60)

# -------------------------- 4. 模型可视化（贴合学术图表风格） --------------------------
plt.figure(figsize=(10, 6))

# 绘制散点图（论文实测数据）
plt.scatter(X, y, color='#2E86AB', s=150, alpha=0.8, edgecolors='#1A5276', linewidth=2, label='Observed Data')

# 绘制拟合直线（覆盖自变量取值范围）
X_range = np.linspace(X.min() - 0.1, X.max() + 0.1, 100).reshape(-1, 1)  # 扩展x轴范围，使直线更完整
y_range_pred = model.predict(X_range)
plt.plot(X_range, y_range_pred, color='#E63946', linewidth=3, alpha=0.9, label=f'Fitted Line (R²={r2:.4f})')

# 标注每个数据点的污染等级
for i in range(len(df)):
    plt.annotate(
        df['Pollution Level'].iloc[i],
        xy=(X[i][0], y[i]),
        xytext=(5, 5),  # 文本偏移量
        textcoords='offset points',
        fontsize=10,
        color='#1A5276',
        fontweight='bold'
    )

# 设置图表标签和标题（贴合学术规范）
plt.xlabel('OH Production Rate (ppbV/h)', fontsize=12, fontweight='bold')
plt.ylabel('Sulfur Oxidation Rate (SOR, %)', fontsize=12, fontweight='bold')
plt.title('Relationship between OH Production Rate and SOR', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')

# 调整坐标轴范围（使数据点分布更合理）
plt.xlim(X.min() - 0.2, X.max() + 0.2)
plt.ylim(y.min() - 0.05, y.max() + 0.05)

plt.tight_layout()
plt.savefig('OHSOR.png', dpi=150, bbox_inches='tight')
# plt.show()  # Non-interactive mode
print("Figure saved as OHSOR.png")

# -------------------------- 5. 模型应用（预测新场景） --------------------------
print("\n[Model Application: Predicting SOR for New Scenarios]")
print("Scenario 1: OH Production Rate = 0.6 ppbV/h (between SP and MP)")
P_OH_test1 = np.array([[0.6]])
SOR_pred1 = model.predict(P_OH_test1)[0]
print(f"Predicted SOR = {SOR_pred1:.4f}%")

print("\nScenario 2: OH Production Rate = 0.8 ppbV/h (between MP and HP)")
P_OH_test2 = np.array([[0.8]])
SOR_pred2 = model.predict(P_OH_test2)[0]
print(f"Predicted SOR = {SOR_pred2:.4f}%")

print("\nScenario 3: OH Production Rate = 1.3 ppbV/h (high activity, beyond data range)")
P_OH_test3 = np.array([[1.3]])
SOR_pred3 = model.predict(P_OH_test3)[0]
print(f"Predicted SOR = {SOR_pred3:.4f}%")
print("="*60)

# ====================== 保存模型 ======================
with open('oh_sor_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as oh_sor_model.pkl")

# ====================== 预测函数 ======================
def predict_sor_from_oh(oh_production_rate_ppbV_h):
    """
    Predict Sulfur Oxidation Rate (SOR) from OH production rate

    Args:
        oh_production_rate_ppbV_h: OH production rate (ppbV/h)

    Returns:
        Sulfur Oxidation Rate SOR (%)
    """
    try:
        with open('oh_sor_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Warning: oh_sor_model.pkl not found, please run igemOHSOR.py to train the model")
        return None

    sor = model.predict([[oh_production_rate_ppbV_h]])[0]
    # Ensure SOR is within reasonable range
    sor = max(0, min(100, sor))
    return sor


if __name__ == "__main__":
    # Test prediction function
    test_oh = 0.8  # ppbV/h
    predicted_sor = predict_sor_from_oh(test_oh)
    print(f"\nTest: OH Production Rate={test_oh} ppbV/h, Predicted SOR={predicted_sor:.2f}%")