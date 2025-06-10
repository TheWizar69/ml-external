### 1. Write a Python program to implement Simple Linear Regression.

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

### 2. Write a Python program to a Decision tree using sklearn and its parameter tuning.

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

### 3. Write a Python program to implement KNN using sklearn.

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

### 4. Write a Python program to implement Logistic Regression using sklearn 

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

### 5. Write a Python program to implement K-Means Clustering

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

### 1. Visualise a PIE CHART using MatPlotLib

```python
import matplotlib.pyplot as plt
import numpy as np

labels = ["Apple", "Banana", "Cherry", "Dates"]
sizes = [20, 30, 25, 25]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['red', 'Yellow', 'Pink', 'Brown'])
plt.title('Pie Chart Example')
plt.show()
```

### 2. Visualise a Histogram using Matplotlib

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(1000)
plt.hist(data, bins=30, color='purple', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram Example')
plt.show()
```

### 3. Visualise a Scatter Plot using Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
y = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])
plt.scatter(x, y)
plt.show()
plt.scatter(x, y, color = 'red')
plt.show()
```
### 4. Visualise a Bar Chart using MatPlotLib

```python
import numpy as np
import matplotlib.pyplot as plt

categories = ['A', 'B', 'C', 'D']
values = [10, 5, 7, 12]
plt.bar(categories, values, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart Example')
plt.show()
```

### 5. Create a Simple Line Plot using labels,grids,titles

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, label = 'sin(x)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')
plt.legend()
plt.grid(True)
plt.show()
```
