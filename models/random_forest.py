from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_random_forest(x_train_encoded, y_train, x_test_encoded, y_test):
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(x_train_encoded, y_train)
    y_pred = rf_classifier.predict(x_test_encoded)
    accuracy = accuracy_score(y_pred, y_test)
    return accuracy
