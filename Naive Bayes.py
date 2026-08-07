import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('titanic.csv')
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features].copy()
y = df['Survived']

# Preprocessing (همون قبلی)
imputer = SimpleImputer(strategy='median')
X[['Age', 'Fare']] = imputer.fit_transform(X[['Age', 'Fare']])
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])

le_sex = LabelEncoder(); X['Sex'] = le_sex.fit_transform(X['Sex'])
le_embarked = LabelEncoder(); X['Embarked'] = le_embarked.fit_transform(X['Embarked'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === اضافه شده: PCA برای کاهش بعد ===
pca = PCA(n_components=0.95)  # یا n_components=2 برای 2 بعد و رسم scatter
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("تعداد ویژگی‌های اصلی:", X_train_scaled.shape[1])
print("تعداد ویژگی‌های PCA:", X_train_pca.shape[1])
print("نسبت واریانس توضیح داده‌شده:", sum(pca.explained_variance_ratio_))

param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
grid = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_pca, y_train)

best_nb = grid.best_estimator_
y_pred = best_nb.predict(X_test_pca)

print("بهترین پارامترها:", grid.best_params_)
print("Accuracy Naive Bayes (با PCA):", accuracy_score(y_test, y_pred))

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Naive Bayes (با PCA)')
plt.show()

