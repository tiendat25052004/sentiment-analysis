from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def split_data(df):
    label_encode = LabelEncoder()
    y_data = label_encode.fit_transform(df['sentiment'])
    x_data = df['review']
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def vectorize_text(x_train, x_test):
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_vectorizer.fit(x_train)
    x_train_encoded = tfidf_vectorizer.transform(x_train)
    x_test_encoded = tfidf_vectorizer.transform(x_test)
    return x_train_encoded, x_test_encoded
