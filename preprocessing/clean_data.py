import re
import string
import nltk
from bs4 import BeautifulSoup
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tải các tài nguyên cần thiết
nltk.download('stopwords')
nltk.download('wordnet')

stop = set(stopwords.words('english'))


def expand_contractions(text):
    return contractions.fix(text)


def preprocess_text(text):
    wl = WordNetLemmatizer()
    soup = BeautifulSoup(text, "html.parser")  # Xoá thẻ HTML
    text = soup.get_text()
    text = expand_contractions(text)  # Mở rộng contractions
    text = re.sub(r'\.(?=\S)', '. ', text)  # Thêm dấu cách sau dấu chấm
    text = re.sub(r'http\S+', '', text)  # Xoá URL
    # Xoá dấu câu
    text = "".join([word.lower()
                   for word in text if word not in string.punctuation])
    text = " ".join([wl.lemmatize(word) for word in text.split()
                    if word not in stop and word.isalpha()])  # Lemmatize
    return text
