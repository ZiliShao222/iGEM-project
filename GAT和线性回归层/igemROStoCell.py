import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
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
print(f"ROS-Cell Viability Regression Model:")
print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R²: {r2:.4f}")

# Save model
with open('ros_cell_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as ros_cell_model.pkl")

# Visualization
plt.figure(figsize=(8, 8))
plt.scatter(ros_conc, cell_activity, s=100, color='blue', label='Experimental data')
plt.plot(ros_conc, y_pred, '--', linewidth=2, label=f'Regression line (R²={r2:.4f})')
plt.xlabel('ROS concentration (μmol/L)', fontsize=12)
plt.ylabel('Cell viability (%)', fontsize=12)
plt.title('Linear Regression: ROS Concentration vs Cell Viability', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ROStoCell.png', dpi=150, bbox_inches='tight')
# plt.show()  # Non-interactive mode
print("Figure saved as ROStoCell.png")


# ====================== 预测函数 ======================
def predict_cell_viability(ros_concentration_umol):
    """
    Predict cell viability from ROS concentration

    Args:
        ros_concentration_umol: ROS concentration (μmol/L)

    Returns:
        Cell viability (%)
    """
    try:
        with open('ros_cell_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Warning: ros_cell_model.pkl not found, using current model")
        model = model

    cell_viab = model.predict([[ros_concentration_umol]])[0]
    # Ensure cell viability is within reasonable range
    cell_viab = max(0, min(100, cell_viab))
    return cell_viab


if __name__ == "__main__":
    # Test prediction function
    test_ros = 25  # μmol/L
    predicted_viability = predict_cell_viability(test_ros)
    print(f"\nTest: ROS={test_ros} μmol/L, Predicted cell viability={predicted_viability:.2f}%")
