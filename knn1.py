import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv('titanic.csv')
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features].copy()
y = df['Survived']


imputer = SimpleImputer(strategy='mean')
X[['Age', 'Fare']] = imputer.fit_transform(X[['Age', 'Fare']])
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])


le_sex = LabelEncoder()
X['Sex'] = le_sex.fit_transform(X['Sex'])

le_embarked = LabelEncoder()
X['Embarked'] = le_embarked.fit_transform(X['Embarked'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



pca = PCA(n_components=4, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


param_grid = {'n_neighbors': range(3, 15), 'weights': ['uniform', 'distance']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_pca, y_train)

best_knn = grid.best_estimator_
y_pred = best_knn.predict(X_test_pca)

print("Best Parameters:", grid.best_params_)
print("Accuracy KNN (with PCA):", accuracy_score(y_test, y_pred))


new_passenger = pd.DataFrame([{
    'Pclass': 1,
    'Sex': 'female',
    'Age': 22,
    'SibSp': 1,
    'Parch':0,
    'Fare': 150,
    'Embarked': 'C'
}])


new_passenger['Sex'] = le_sex.transform(new_passenger['Sex'])
new_passenger['Embarked'] = le_embarked.transform(new_passenger['Embarked'])


new_passenger[['Age', 'Fare']] = imputer.transform(new_passenger[['Age', 'Fare']])


new_passenger_scaled = scaler.transform(new_passenger)
new_passenger_pca = pca.transform(new_passenger_scaled)


prediction = best_knn.predict(new_passenger_pca)

print("\n--- Prediction for New Passenger ---")
if prediction[0] == 1:
    print("نتیجه: زنده می‌ماند (Survived)")
else:
    print("نتیجه: فوت می‌کند (Not Survived)")


plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - KNN with PCA')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
