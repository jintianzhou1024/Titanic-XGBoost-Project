import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

# --- 1. 读取数据 ---
print("正在尝试读取数据...")

# 检查文件是否存在，防止路径报错
if not os.path.exists('train.csv'):
    print("错误：找不到 train.csv 文件！")
    print(f"请确保 train.csv 和代码在同一个文件夹里。")
    print(f"当前代码运行目录是: {os.getcwd()}")
    exit()

df = pd.read_csv('train.csv')
print("成功读取数据！")

# --- 2. 数据清洗 (Data Cleaning) ---
# 选取关键特征
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']
df = df[features]

# 填补年龄缺失值 (用中位数)
df['Age'] = df['Age'].fillna(df['Age'].median())
# 删掉剩下的缺失值
df.dropna(inplace=True)

# 将 'Sex' (male/female) 转换成数字
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# --- 3. 准备训练数据 ---
X = df.drop('Survived', axis=1)  # 特征
y = df['Survived']               # 目标

# 拆分数据 (80%训练, 20%测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. 训练 XGBoost ---
print("正在训练 XGBoost 模型...")
# 为了兼容性，如果你没装新版库，use_label_encoder=False 可以防止报错
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train, y_train)

# --- 5. 预测结果 ---
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n" + "="*40)
print(f"恭喜！模型运行成功！")
print(f"预测准确率 (Accuracy): {accuracy * 100:.2f}%")
print("="*40 + "\n")

# 显示特征重要性
print("特征重要性排序:")
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))