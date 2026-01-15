import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import warnings

# 忽略一些版本警告，保持界面干净
warnings.filterwarnings('ignore')

# --- 1. 读取数据 ---
if not os.path.exists('train.csv'):
    print("找不到 train.csv！")
    exit()
df = pd.read_csv('train.csv')

# --- 2. 特征工程 (保持之前的高级版) ---
df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
                                   'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 1
df['IsAlone'].loc[df['FamilySize'] > 1] = 0

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# 处理 Fare (票价) 的缺失值
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# 数字化
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])
df['Title'] = le.fit_transform(df['Title'])

# 准备数据
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

# --- 3. 关键步骤：GridSearchCV 自动调参 ---
print("正在启动自动调参 (Grid Search)... 这可能需要十几秒...")

# 我们给模型一组备选参数
param_grid = {
    'n_estimators': [100, 200, 300],    # 树的数量
    'learning_rate': [0.01, 0.05, 0.1], # 学习快慢
    'max_depth': [3, 4, 5],             # 树的深度
    'min_child_weight': [1, 3],         # 防止过拟合的参数
    'gamma': [0, 0.1]                   # 另一个防止过拟合的参数
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# cv=5 表示做5折交叉验证 (数据切5份轮流考)，最严谨的验证方法
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

# --- 4. 输出结果 ---
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n" + "="*50)
print(f"调参结束！找到的最佳参数是: {grid_search.best_params_}")
print("-" * 50)
print(f"最终测试集准确率 (Accuracy): {accuracy * 100:.2f}%")
print("="*50 + "\n")