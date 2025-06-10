### Program to implement single layer perceptron for AND Gate.

```python
import numpy as np

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_outputs = np.array([0, 0, 0, 1])
weights = np.array([0.3, -0.2])
bias = -0.4
learning_rate = 0.2

def predict(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    return 1 if weighted_sum >= 0 else 0

def train_perceptron(inputs, target_outputs, weights, bias, learning_rate, epochs=4):
    weights = weights.copy()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        
        for i in range(len(inputs)):
            prediction = predict(inputs[i], weights, bias)
            error = target_outputs[i] - prediction
            
            if error != 0:
                print(f"Update for input {inputs[i]}:")
                print(f"Predicted : {prediction}, Target : {target_outputs[i]}")
                print(f"Error : {error}")
                
                weights += learning_rate * error * inputs[i]
                bias += learning_rate * error
                
                print(f"Updated weights : {weights}")
                print(f"Updated bias : {bias}")
            else:
                print(f"No update for input {inputs[i]} (prediction is correct)")
        
        print("-" * 40)
    
    return weights, bias

final_weights, final_bias = train_perceptron(inputs, target_outputs, weights, bias, learning_rate)

print(f"Final weights : {final_weights}")
print(f"Final bias : {final_bias}")

print("Testing after training:")
for i in range(len(inputs)):
    prediction = predict(inputs[i], final_weights, final_bias)
    print(f"Input : {inputs[i]} => Predicted output: {prediction}, Target: {target_outputs[i]}")
```

### Implementation of Multilayer Perceptron for XOR Gate

```python
import numpy as np
from sklearn.neural_network import MLPClassifier

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

mlp = MLPClassifier(hidden_layer_sizes=(3,), activation='relu', solver='adam', max_iter=2000, random_state=42)
mlp.fit(x, y)

predictions = mlp.predict(x)
print("\nXOR Gate prediction using MLP:")

for i in range(len(x)):
    print(f"Input : {x[i]} => Predicted output : {predictions[i]}, Target : {y[i]}")

accuracy = mlp.score(x, y)
print(f"\nModel Accuracy : {accuracy * 100:.2f}%")
```

### Implement a program to perform Simple Linear Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([8, 6, 4, 2]).reshape(-1, 1)
y = np.array([3, 4, 5, 3])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)

plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, y_pred, color='red', label="Regression Line")

plt.xlabel("X values")
plt.ylabel("Y values")
plt.legend()
plt.show()
```

### Implementation of Multiple Linear Regression for House Price Prediction using sklearn

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("houseprice.csv")
print(df)
df = df.dropna()
X = df[["Size_in_sqft", "Bedrooms", "House_Age"]]
y = df["Price_in_lakhs"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
new_house = pd.DataFrame([[2000, 3, 5]], columns=["Size_in_sqft", "Bedrooms", "House_Age"])
predicted_price = model.predict(new_house)
print(f"Predicted Price for New House: {predicted_price[0]:.2f} Lakhs")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.5, label="Predicted")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="dashed", label="Ideal Fit")
plt.xlabel("Actual prices (Lakhs)")
plt.ylabel("Predicted prices (Lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()
```

### houseprice.csv

```python
Size_in_sqft,Bedrooms,House_Age,Price_in_lakhs
850,2,10,22
900,2,15,25
1100,3,8,30
1400,3,5,38
1600,4,7,45
1800,4,12,50
2000,4,3,55
2200,5,2,60
2500,5,1,70
2700,6,4,75
```

### Implementation of a Decision Tree using sklearn and its parameters tuning

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier()
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
plt.figure(figsize=(12, 8))
plot_tree(best_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```

### Implementation of KNN using sklearn

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - KNN')
plt.show()
```

### Implementation of Logistic Regression using sklearn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris 

iris = load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()
plot_decision_boundary(X_train, y_train, model)
```

### Implementation of K-Means clustering

```python
import numpy as np
from matplotlib import pyplot as plt

X = np.random.randint(10,35,(25,2))
Y = np.random.randint(55,70,(25,2))
Z = np.vstack((X,Y))
Z = Z.reshape((50,2))
Z = np.float32(Z)
plt.scatter(Z[:, 0], Z[:, 1], c='blue', edgecolors='k', label="Data Points")
plt.xlabel('Test Data')
plt.ylabel('Z samples')
plt.title("Scatter Plot of Z Samples")
plt.legend()
plt.show()

```

### Performance analysis of Classification Algorithms on a specific dataset. ( Logistic Regression,SVM,  Random Forest)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier()
}
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("-" * 50)
plt.figure(figsize=(10,5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.title("Comparison of Classification Algorithms on Iris Dataset")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,6))
for name, clf in classifiers.items():
    y_score = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
    if y_score is not None:
        for i in range(3):  # Iris has 3 classes
            fpr, tpr, _ = roc_curve(y_test == i, y_score[:, i])
            plt.plot(fpr, tpr, label=f"{name} (class {i})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
```
