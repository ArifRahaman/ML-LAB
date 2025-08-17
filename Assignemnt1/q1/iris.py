import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report



iris_data = pd.read_csv("iris_dataset.csv")
X = iris_data .drop('target', axis=1)
y = iris_data ['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


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


mnb = MultinomialNB(alpha=0.4,force_alpha=True)
mnb.fit(X_train, y_train)
evaluate_model("MultinomialNB", y_test, mnb.predict(X_test))


bnb = BernoulliNB(alpha=0.4)
bnb.fit(X_train, y_train)
evaluate_model("BernoulliNB", y_test, bnb.predict(X_test))


gnb = GaussianNB()
gnb.fit(X_train, y_train)
evaluate_model("GaussianNB", y_test, gnb.predict(X_test))
