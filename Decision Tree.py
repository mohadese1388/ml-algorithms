import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

# بارگذاری داده
df = pd.read_csv('titanic.csv')
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features].copy()
y = df['Survived']

# --- اصلاح پیش‌پردازش ---
# پر کردن مقادیر گمشده عددی
imputer = SimpleImputer(strategy='median')
X[['Age', 'Fare']] = imputer.fit_transform(X[['Age', 'Fare']])

# پر کردن مقادیر گمشده دسته‌ای (Categorical)
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])

# تبدیل متغیرهای متنی به عدد
le_sex = LabelEncoder()
X['Sex'] = le_sex.fit_transform(X['Sex'])

le_embarked = LabelEncoder()
X['Embarked'] = le_embarked.fit_transform(X['Embarked'])

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# استانداردسازی (بسیار مهم برای PCA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- PCA ---
# استفاده از 0.95 یعنی حفظ 95 درصد واریانس داده‌ها
pca = PCA(n_components=0.95) 
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"تعداد ویژگی‌های اصلی: {X_train_scaled.shape[1]}")
print(f"تعداد ویژگی‌های PCA: {X_train_pca.shape[1]}")
print(f"نسبت واریانس توضیح داده‌شده: {sum(pca.explained_variance_ratio_):.2f}")

# --- مدل‌سازی و GridSearchCV ---
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_pca, y_train)

best_dt = grid.best_estimator_()
y_pred = best_dt.predict(X_test_pca)

print("بهترین پارامترها:", grid.best_params_)
print(f"Accuracy Decision Tree (با PCA): {accuracy_score(y_test, y_pred):.4f}")

# --- رسم Confusion Matrix ---
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Decision Tree (با PCA)')
plt.show()

