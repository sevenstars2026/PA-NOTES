"""
对数几率回归（Logistic Regression）算法实现
核心原理：
- sigmoid函数：σ(z) = 1 / (1 + exp(-z))，将线性输出映射到[0,1]
- 预测概率：P(y=1|x) = σ(w·x + b)
- 损失函数：交叉熵损失 L = -Σ[y·ln(ŷ) + (1-y)·ln(1-ŷ)]
- 优化方法：梯度下降法
"""

import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        """sigmoid激活函数：σ(z) = 1/(1+exp(-z))"""
        z = np.clip(z, -500, 500)  # 防止溢出
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, X, y, predictions):
        """计算交叉熵损失"""
        m = X.shape[0]
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -1/m * np.sum(y * np.log(predictions) + 
                             (1 - y) * np.log(1 - predictions))
        return loss

    def fit(self, X, y):
        """
        梯度下降训练过程：
        1. 初始化 w=0, b=0
        2. 循环迭代：
           - 前向：z = w·x + b, ŷ = sigmoid(z)
           - 计算损失
           - 反向：∂L/∂w = -X.T·(y-ŷ)/m
           - 更新：w := w - lr·∂L/∂w
        """
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for iteration in range(self.n_iterations):
            # 前向传播
            z = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(z)

            # 计算损失
            loss = self._compute_loss(X, y, predictions)
            self.loss_history.append(loss)

            # 计算梯度
            dw = -1/m * np.dot(X.T, (y - predictions))
            db = -1/m * np.sum(y - predictions)

            # 更新参数
            self.weights -= self.learning_rate * dw
            if self.fit_intercept:
                self.bias -= self.learning_rate * db

            if (iteration + 1) % 200 == 0:
                print(f"迭代 {iteration + 1}/{self.n_iterations}, 损失: {loss:.4f}")

        return self

    def predict_proba(self, X):
        """预测正例概率"""
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X):
        """预测标签（概率>=0.5为1，否则为0）"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
