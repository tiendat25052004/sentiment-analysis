from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def train_decision_tree(x_train_encoded, y_train, x_test_encoded, y_test):
    dt_classifier = DecisionTreeClassifier(
        criterion='entropy', random_state=42)
    dt_classifier.fit(x_train_encoded, y_train)
    y_pred = dt_classifier.predict(x_test_encoded)
    accuracy = accuracy_score(y_pred, y_test)
    return accuracy
