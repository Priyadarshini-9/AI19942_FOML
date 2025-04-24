import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create DataFrame for pairplot before PCA
df_original = pd.DataFrame(X, columns=feature_names)
df_original['Target'] = [target_names[i] for i in y]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create PCA DataFrame for plotting
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Target'] = [target_names[i] for i in y]

# -------------------------------------------
# Train-test split & classifier for accuracy
# -------------------------------------------
# Original data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_original = clf.predict(X_test)
acc_original = accuracy_score(y_test, y_pred_original)

# PCA-reduced data
X_pca_train, X_pca_test, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)
clf_pca = LogisticRegression()
clf_pca.fit(X_pca_train, y_train)
y_pred_pca = clf_pca.predict(X_pca_test)
acc_pca = accuracy_score(y_test, y_pred_pca)

# -------------------------------------------
# Print Accuracy
# -------------------------------------------
print("Accuracy on Original Data: {:.2f}%".format(acc_original * 100))
print("Accuracy after PCA (2D): {:.2f}%".format(acc_pca * 100))

# -------------------------------------------
# Visualizations
# -------------------------------------------

# 1. Pairplot of original features
sns.pairplot(df_original, hue='Target', corner=True)
plt.suptitle("Pairplot of Original Features", y=1.02)
plt.show()

# 2. PCA scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Target', palette='Set2')
plt.title('PCA - 2D Representation of Iris Dataset')
plt.grid(True)
plt.show()

# 3. Explained variance
plt.figure(figsize=(6,4))
sns.barplot(x=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            y=pca.explained_variance_ratio_)
plt.title("Explained Variance by Each Principal Component")
plt.ylabel("Proportion of Variance")
plt.show()

# 4. Cumulative variance
pca_full = PCA().fit(X_scaled)
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o', color='purple')
plt.axhline(y=0.95, color='red', linestyle='--', label='95% variance')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.grid(True)
plt.legend()
plt.show()

# 5. Heatmap of PCA Components
plt.figure(figsize=(8,4))
sns.heatmap(pca.components_, cmap='coolwarm', annot=True,
            xticklabels=feature_names, yticklabels=['PC1', 'PC2'])
plt.title("PCA Component Loadings")
plt.show()
