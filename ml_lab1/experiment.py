"""
数据加载、预处理和实验脚本
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression


def load_iris_data():
    """加载鸢尾花数据集（二分类：0 vs 1）"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 取前两类做二分类问题
    mask = y < 2
    X = X[mask]
    y = y[mask]
    
    return X, y, iris.feature_names, iris.target_names[:2]


def load_watermelon_data():
    """
    加载西瓜数据集（简化版）
    使用两个特征：密度和含糖量
    标签：1=好瓜，0=坏瓜
    """
    # 简化的西瓜数据集
    data = {
        '密度': [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243],
        '含糖量': [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267],
        '好瓜': [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    X = df[['密度', '含糖量']].values
    y = df['好瓜'].values
    
    return X, y, ['密度', '含糖量'], ['坏瓜', '好瓜']


def standardize_features(X_train, X_test):
    """
    特征标准化（很重要！）
    原因：不同特征的量纲和数值范围差异大，会影响学习率的设置
    方法：(x - mean) / std
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(model, X_test, y_test, dataset_name):
    """评估模型性能"""
    print(f"\n{'='*50}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*50}")
    
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    
    print(f"\n准确率: {accuracy:.4f}")
    print(f"\n混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n分类报告:\n{classification_report(y_test, y_pred)}")
    
    return accuracy


def plot_results(model, X_train, y_train, X_test, y_test, feature_names, dataset_name):
    """绘制决策边界和损失曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：损失曲线
    ax1 = axes[0]
    ax1.plot(model.loss_history)
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('损失值')
    ax1.set_title(f'{dataset_name} - 训练损失曲线')
    ax1.grid(True, alpha=0.3)
    
    # 右图：数据点和决策边界
    ax2 = axes[1]
    
    # 绘制训练和测试数据
    ax2.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
               c='blue', marker='o', label='类别0(训练)')
    ax2.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
               c='red', marker='o', label='类别1(训练)')
    ax2.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], 
               c='blue', marker='x', s=100, label='类别0(测试)')
    ax2.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], 
               c='red', marker='x', s=100, label='类别1(测试)')
    
    # 绘制决策边界（对于2维特征）
    if X_train.shape[1] == 2:
        h = 0.02
        x_min, x_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
        y_min, y_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax2.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
        ax2.contour(xx, yy, Z, colors='black', linewidths=0.5)
    
    ax2.set_xlabel(feature_names[0])
    ax2.set_ylabel(feature_names[1] if len(feature_names) > 1 else '特征2')
    ax2.set_title(f'{dataset_name} - 决策边界')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_results.png', dpi=100)
    print(f"\n图表已保存为: {dataset_name}_results.png")
    plt.close()


def main():
    print("\n" + "="*60)
    print("对数几率回归实验")
    print("="*60)
    
    # 实验1：鸢尾花数据集
    print("\n【实验1：鸢尾花数据集】")
    X_iris, y_iris, iris_features, iris_names = load_iris_data()
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=42
    )
    X_train_iris_scaled, X_test_iris_scaled, _ = standardize_features(
        X_train_iris, X_test_iris
    )
    
    model_iris = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    model_iris.fit(X_train_iris_scaled, y_train_iris)
    evaluate_model(model_iris, X_test_iris_scaled, y_test_iris, "鸢尾花数据集")
    plot_results(model_iris, X_train_iris_scaled, y_train_iris, 
                X_test_iris_scaled, y_test_iris, iris_features[:2], "iris")
    
    # 实验2：西瓜数据集
    print("\n【实验2：西瓜数据集】")
    X_melon, y_melon, melon_features, melon_names = load_watermelon_data()
    X_train_melon, X_test_melon, y_train_melon, y_test_melon = train_test_split(
        X_melon, y_melon, test_size=0.3, random_state=42
    )
    X_train_melon_scaled, X_test_melon_scaled, _ = standardize_features(
        X_train_melon, X_test_melon
    )
    
    model_melon = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    model_melon.fit(X_train_melon_scaled, y_train_melon)
    evaluate_model(model_melon, X_test_melon_scaled, y_test_melon, "西瓜数据集")
    plot_results(model_melon, X_train_melon_scaled, y_train_melon,
                X_test_melon_scaled, y_test_melon, melon_features, "melon")
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)

if __name__ == '__main__':
    main()
