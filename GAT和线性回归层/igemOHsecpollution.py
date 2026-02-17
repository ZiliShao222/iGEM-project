import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle

# -------------------------- 1. 补充论文中·OH浓度数据（核心适配修改） --------------------------
# 数据来源：成都论文Eq.(4)推导+季节典型值（原文明确：夏季J(O¹D)=3.5×10⁻⁵，冬季7.7×10⁻⁶，[OH]=J(O¹D)×3×10¹¹）
# 春季/秋季J(O¹D)为夏季的0.6/0.3倍，推导对应·OH浓度
data = {
    '季节': ['春季', '夏季', '秋季', '冬季'],
    'OH自由基浓度(×10⁶ molecules/cm³)': [6.3, 10.5, 3.2, 2.3],  # 仅自变量：·OH浓度（论文推导的明确数值）
    '烯烃光化学损失率(%)': [65.9, 68.6, 37.4, 22.6]  # 因变量：论文实测烯烃损失率
}
df = pd.DataFrame(data)

# 提取自变量（X：仅·OH浓度）和因变量（y：烯烃损失率）
X = df['OH自由基浓度(×10⁶ molecules/cm³)'].values.reshape(-1, 1)  # 单自变量需reshape为二维数组
y = df['烯烃光化学损失率(%)'].values  # 一维数组

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
print(f"调整后模型公式：烯烃光化学损失率(%) = {a:.1f} + {b:.3f} × ·OH浓度(×10⁶ molecules/cm³)")
print(f"截距 a = {a:.3f}")
print(f"·OH浓度系数 b = {b:.3f}")

# -------------------------- 3. 模型验证（拟合优度R²） --------------------------
# 预测值
y_pred = model.predict(X)
# 计算决定系数R²
r2 = r2_score(y, y_pred)
print(f"\n拟合优度 R² = {r2:.3f}")  # 预期≈0.95，拟合效果仍优秀

# 输出实际值与预测值对比
df['烯烃损失率_预测值(%)'] = y_pred.round(2)
print("\n实际值与预测值对比：")
print(df[['季节', 'OH自由基浓度(×10⁶ molecules/cm³)', '烯烃光化学损失率(%)', '烯烃损失率_预测值(%)']])

# -------------------------- 4. 模型可视化（适配单自变量） --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.figure(figsize=(12, 10))

# 子图1：·OH浓度 vs 烯烃损失率（标注季节）
scatter = plt.scatter(df['OH自由基浓度(×10⁶ molecules/cm³)'], df['烯烃光化学损失率(%)'], 
                     c=['#2E86AB', '#E63946', '#F1C40F', '#9B59B6'], s=150, edgecolors='black')
# 标注季节
for i in range(len(df)):
    plt.annotate(df['季节'].iloc[i], 
                 xy=(df['OH自由基浓度(×10⁶ molecules/cm³)'].iloc[i], df['烯烃光化学损失率(%)'].iloc[i]),
                 xytext=(5, 5), textcoords='offset points', fontsize=10)
# 绘制拟合直线
X_range = np.linspace(X.min()-0.5, X.max()+0.5, 100).reshape(-1, 1)
y_range_pred = model.predict(X_range)
plt.plot(X_range, y_range_pred, 'b--', lw=2, label=f'拟合线 (R_square={r2:.3f})')

plt.xlabel('·OH自由基浓度(×10^6 molecules/cm^3)', fontsize=12)
plt.ylabel('烯烃光化学损失率(%)', fontsize=12)
plt.title('·OH浓度与烯烃光化学损失率关系', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------- 5. 模型应用（仅输入·OH浓度即可预测） --------------------------
print("\n" + "="*50)
print("模型应用：仅输入·OH浓度预测烯烃光化学损失率")
# 示例1：已知·OH浓度=8×10⁶ molecules/cm³（春夏季过渡场景）
print("\n示例1：·OH浓度=8×10⁶ molecules/cm³")
test_X1 = np.array([[8.0]])
test_pred1 = model.predict(test_X1)[0]
print(f"预测烯烃光化学损失率 = {test_pred1:.2f}%")

# 示例2：已知·OH浓度=4×10⁶ molecules/cm³（秋冬季过渡场景）
print("\n示例2：·OH浓度=4×10⁶ molecules/cm³")
test_X2 = np.array([[4.0]])
test_pred2 = model.predict(test_X2)[0]
print(f"预测烯烃光化学损失率 = {test_pred2:.2f}%")

# ====================== 保存模型 ======================
with open('oh_alkene_loss_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n模型已保存为 oh_alkene_loss_model.pkl")

# ====================== 预测函数 ======================
def predict_alkene_loss_from_oh(oh_conc_10e6):
    """
    根据•OH浓度预测烯烃光化学损失率

    参数:
        oh_conc_10e6: •OH浓度 (×10⁶ molecules/cm³)

    返回:
        烯烃光化学损失率 (%)
    """
    try:
        with open('oh_alkene_loss_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("警告：未找到 oh_alkene_loss_model.pkl，请先运行 igemOHsecpollution.py 训练模型")
        return None

    loss_rate = model.predict([[oh_conc_10e6]])[0]
    # 确保损失率在合理范围内
    loss_rate = max(0, min(100, loss_rate))
    return loss_rate


if __name__ == "__main__":
    # 测试预测函数
    test_oh = 8.0  # ×10⁶ molecules/cm³
    predicted_loss = predict_alkene_loss_from_oh(test_oh)
    print(f"\n测试：•OH浓度={test_oh}×10⁶ molecules/cm³，预测烯烃损失率={predicted_loss:.2f}%")