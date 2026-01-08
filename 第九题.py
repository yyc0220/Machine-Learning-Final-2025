import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# 1. 加载数据
file_path = '回归预测.xlsx'
try:
    # 按照题目要求，Sheet1训练，Sheet2测试
    train_df = pd.read_excel(file_path, sheet_name=0, header=None)
    test_df = pd.read_excel(file_path, sheet_name=1, header=None)
    print("数据读取成功")
except Exception as e:
    print(f"数据读取失败: {e}")

# 2. 数据预处理
# 为保证One-hot编码的一致性，合并处理
train_df['source'] = 'train'
test_df['source'] = 'test'
combined = pd.concat([train_df, test_df], axis=0)

# 对分类特征（药物名称，列索引30）进行独热编码
combined = pd.get_dummies(combined, columns=[30])

# 还原回训练集和测试集
train_proc = combined[combined['source'] == 'train'].drop('source', axis=1)
test_proc = combined[combined['source'] == 'test'].drop('source', axis=1)

# X 为特征，y 为最后一列（列索引31）
X_train = train_proc.drop(31, axis=1)
y_train = train_proc[31]
X_test = test_proc.drop(31, axis=1)
y_test = test_proc[31]

# 将列名统一转为字符串（scikit-learn规范）
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# 3. 自定义评分函数：为了让CV过程更贴合题目要求的“平方和相对误差”
def ssre_scorer(y_true, y_pred):
    # 平方和相对误差的负值（因为网格搜索默认选择得分最高的，负值越小意味着误差越小）
    error = np.sum(((y_true - y_pred) / (y_true + 1e-9))**2)
    return -error

custom_score = make_scorer(ssre_scorer, greater_is_better=True)

# 4. 集成学习：随机森林 + 交叉验证
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(rf, param_grid, cv=5, scoring=custom_score, n_jobs=-1)
grid.fit(X_train, y_train)

# 5. 预测与评估报告
print(f"最优参数组合: {grid.best_params_}")
y_pred = grid.best_estimator_.predict(X_test)

# 计算每个测试样本的相对平方误差
individual_relative_errors = ((y_test - y_pred) / (y_test + 1e-9))**2

print(f"测试集样本误差均值 (Mean): {individual_relative_errors.mean():.6f}")
print(f"测试集样本误差方差 (Variance): {individual_relative_errors.var():.6f}")
print(f"测试集总平方和相对误差 (SSRE): {individual_relative_errors.sum():.6f}")
