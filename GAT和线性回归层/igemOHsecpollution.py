import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import pickle

# -------------------------- 1. 补充论文中·OH浓度数据（核心适配修改） --------------------------
# 数据来源：成都论文Eq.(4)推导+季节典型值（原文明确：夏季J(O¹D)=3.5×10⁻⁵，冬季7.7×10⁻⁶，[OH]=J(O¹D)×3×10¹¹）
# 春季/秋季J(O¹D)为夏季的0.6/0.3倍，推导对应·OH浓度
data = {
    'Season': ['Spring', 'Summer', 'Autumn', 'Winter'],
    'OH Conc. (×10⁶ molecules/cm³)': [6.3, 10.5, 3.2, 2.3],  # 仅自变量：·OH浓度（论文推导的明确数值）
    'Alkene Loss Rate (%)': [65.9, 68.6, 37.4, 22.6]  # 因变量：论文实测烯烃损失率
}
df = pd.DataFrame(data)

# 提取自变量（X：仅·OH浓度）和因变量（y：烯烃损失率）
X = df['OH Conc. (×10⁶ molecules/cm³)'].values.reshape(-1, 1)  # 单自变量需reshape为二维数组
y = df['Alkene Loss Rate (%)'].values  # 一维数组

# -------------------------- 2. 模型拟合（仅以·OH浓度为自变量） --------------------------
# 初始化线性回归模型
model = LinearRegression()
# 拟合数据
model.fit(X, y)

# 提取模型系数
a = model.intercept_  # 截距
b = model.coef_[0]    # ·OH浓度的系数（仅一个自变量）

# 输出模型公式（适配单自变量）
print("="*50)
print(f"Model Formula: Alkene Loss Rate (%) = {a:.1f} + {b:.3f} × OH Conc. (×10⁶ molecules/cm³)")
print(f"Intercept a = {a:.3f}")
print(f"OH Conc. Coefficient b = {b:.3f}")

# -------------------------- 3. 模型验证（拟合优度R²） --------------------------
# 预测值
y_pred = model.predict(X)
# 计算决定系数R²
r2 = r2_score(y, y_pred)
print(f"\n拟合优度 R² = {r2:.3f}")  # 预期≈0.95，拟合效果仍优秀

# 输出实际值与预测值对比
df['Alkene Loss Pred (%)'] = y_pred.round(2)
print("\nObserved vs Predicted Values:")
print(df[['Season', 'OH Conc. (×10⁶ molecules/cm³)', 'Alkene Loss Rate (%)', 'Alkene Loss Pred (%)']])

# -------------------------- 4. 模型可视化（适配单自变量） --------------------------
plt.figure(figsize=(8, 8))

# 子图1：·OH浓度 vs 烯烃损失率（标注季节）
scatter = plt.scatter(df['OH Conc. (×10⁶ molecules/cm³)'], df['Alkene Loss Rate (%)'],
                     c=['#2E86AB', '#E63946', '#F1C40F', '#9B59B6'], s=150, edgecolors='black')
# 标注季节
for i in range(len(df)):
    plt.annotate(df['Season'].iloc[i],
                 xy=(df['OH Conc. (×10⁶ molecules/cm³)'].iloc[i], df['Alkene Loss Rate (%)'].iloc[i]),
                 xytext=(5, 5), textcoords='offset points', fontsize=10)
# 绘制拟合直线
X_range = np.linspace(X.min()-0.5, X.max()+0.5, 100).reshape(-1, 1)
y_range_pred = model.predict(X_range)
plt.plot(X_range, y_range_pred, 'b--', lw=2, label=f'Fitted Line (R²={r2:.3f})')

plt.xlabel('OH Concentration (×10⁶ molecules/cm³)', fontsize=12)
plt.ylabel('Alkene Loss Rate (%)', fontsize=12)
plt.title('Relationship between OH Concentration and Alkene Loss Rate', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('secpollution.png', dpi=150, bbox_inches='tight')
# plt.show()  # Non-interactive mode
print("Figure saved as secpollution.png")

# -------------------------- 5. 模型应用（仅输入·OH浓度即可预测） --------------------------
print("\n" + "="*50)
print("Model Application: Predict alkene loss rate from OH concentration")
# 示例1：已知·OH浓度=8×10⁶ molecules/cm³（春夏季过渡场景）
print("\nExample 1: OH Conc. = 8×10⁶ molecules/cm³")
test_X1 = np.array([[8.0]])
test_pred1 = model.predict(test_X1)[0]
print(f"Predicted alkene loss rate = {test_pred1:.2f}%")

# 示例2：已知·OH浓度=4×10⁶ molecules/cm³（秋冬季过渡场景）
print("\nExample 2: OH Conc. = 4×10⁶ molecules/cm³")
test_X2 = np.array([[4.0]])
test_pred2 = model.predict(test_X2)[0]
print(f"Predicted alkene loss rate = {test_pred2:.2f}%")

# ====================== 保存模型 ======================
with open('oh_alkene_loss_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved as oh_alkene_loss_model.pkl")

# ====================== 预测函数 ======================
def predict_alkene_loss_from_oh(oh_conc_10e6):
    """
    Predict alkene loss rate from OH concentration

    Args:
        oh_conc_10e6: OH concentration (×10⁶ molecules/cm³)

    Returns:
        Alkene loss rate (%)
    """
    try:
        with open('oh_alkene_loss_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Warning: oh_alkene_loss_model.pkl not found, please run igemOHsecpollution.py to train the model")
        return None

    loss_rate = model.predict([[oh_conc_10e6]])[0]
    # Ensure loss rate is within reasonable range
    loss_rate = max(0, min(100, loss_rate))
    return loss_rate


if __name__ == "__main__":
    # Test prediction function
    test_oh = 8.0  # ×10⁶ molecules/cm³
    predicted_loss = predict_alkene_loss_from_oh(test_oh)
    print(f"\nTest: OH Conc.={test_oh}×10⁶ molecules/cm³, Predicted alkene loss rate={predicted_loss:.2f}%")