import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def evaluate_model(name, y_true, y_pred):
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))

    print("Precision:", precision_score(y_true, y_pred, average='macro', zero_division=0))

    print("Recall:", recall_score(y_true, y_pred, average='macro', zero_division=0))

    print("F1 Score:", f1_score(y_true, y_pred, average='macro', zero_division=0))

    print("Classification Report:\n", classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


cancer = pd.read_csv("breast_cancer_dataset.csv")
X = cancer.drop('target', axis=1)
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



dt_gini = DecisionTreeClassifier(criterion='gini',   random_state=42)
dt_gini.fit(X_train, y_train)
evaluate_model("Iris (Gini)", y_test, dt_gini.predict(X_test))


plt.figure(figsize=(25, 12))  # make the tree wider and taller
plot_tree(
    dt_gini,
    feature_names=X.columns,
    class_names=[str(c) for c in dt_gini.classes_],
    filled=True,
    fontsize=12   # adjust font size
)


plt.show()



dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)
evaluate_model("Iris (Entropy)", y_test, dt_entropy.predict(X_test))


plt.figure(figsize=(25, 12))  # make the tree wider and taller
plot_tree(
    dt_entropy,   # <-- use dt_entropy here
    feature_names=X.columns,
    class_names=[str(c) for c in dt_entropy.classes_],  # <-- also match to dt_entropy
    filled=True,
    fontsize=12   # adjust font size
)
plt.show()



