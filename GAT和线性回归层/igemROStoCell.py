import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

# ====================== 训练ROS-细胞活性线性回归模型 ======================
# ROS reference concentration (μmol/L): converted from fold of control (control 1.0 = 20 μmol/L)
ros_conc = np.array([20, 24, 28, 34])
# Cell viability (%)
cell_activity = np.array([100, 89.0, 84.0, 77.3])

# Reshape to 2D array
X = ros_conc.reshape(-1, 1)
y = cell_activity

# Linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Calculate R²
r2 = model.score(X, y)

# Print results
print(f"ROS-细胞活性回归模型:")
print(f"斜率 (Slope): {model.coef_[0]:.4f}")
print(f"截距 (Intercept): {model.intercept_:.4f}")
print(f"R²: {r2:.4f}")

# Save model
with open('ros_cell_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("模型已保存为 ros_cell_model.pkl")

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(ros_conc, cell_activity, s=100, color='blue', label='Experimental data')
plt.plot(ros_conc, y_pred, '--', linewidth=2, label=f'Regression line (R²={r2:.4f})')
plt.xlabel('ROS concentration (μmol/L)', fontsize=12)
plt.ylabel('Cell viability (%)', fontsize=12)
plt.title('Linear Regression: ROS Concentration vs Cell Viability', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ros_cell_regression.png', dpi=150)
plt.show()
print("Figure saved as ros_cell_regression.png")


# ====================== 预测函数 ======================
def predict_cell_viability(ros_concentration_umol):
    """
    根据ROS浓度（μmol/L）预测细胞活性（%）

    参数:
        ros_concentration_umol: ROS浓度 (μmol/L)

    返回:
        细胞活性 (%)
    """
    try:
        with open('ros_cell_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("警告：未找到 ros_cell_model.pkl，使用刚训练的模型")
        model = model

    cell_viab = model.predict([[ros_concentration_umol]])[0]
    # 确保细胞活性在合理范围内
    cell_viab = max(0, min(100, cell_viab))
    return cell_viab


if __name__ == "__main__":
    # 测试预测函数
    test_ros = 25  # μmol/L
    predicted_viability = predict_cell_viability(test_ros)
    print(f"\n测试：ROS={test_ros} μmol/L，预测细胞活性={predicted_viability:.2f}%")
