import time
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix

# 1. 加载题目要求的 Iris 数据集
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 2. 数据标准化（神经网络对量纲非常敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 3. 定义对比模型
models = {
    "SVM (Linear Kernel)": SVC(kernel='linear', C=1.0),
    "SVM (RBF Kernel)": SVC(kernel='rbf', C=1.0, gamma='scale'),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=2000, random_state=42)
}

print(f"{'模型名称':<25} | {'测试集准确率':<10} | {'5折交叉验证均值':<12} | {'训练耗时(s)':<12}")
print("-" * 75)

for name, model in models.items():
    # 交叉验证评价稳定性
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)

    # 训练耗时统计
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    # 预测
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"{name:<25} | {acc:<12.2%} | {cv_scores.mean():<14.2%} | {train_time:<12.6f}")


for name, model in models.items():
    print(f" {name} 详细评估报告 ")
    y_pred = model.predict(X_test)

    # 打印分类报告（Precision, Recall, F1）
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # 打印混淆矩阵，直观查看误分类情况
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))