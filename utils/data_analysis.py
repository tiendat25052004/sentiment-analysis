import matplotlib.pyplot as plt
import seaborn as sns


def plot_sentiment_distribution(df):
    freq_pos = len(df[df['sentiment'] == 'positive'])
    freq_neg = len(df[df['sentiment'] == 'negative'])
    data = [freq_pos, freq_neg]
    labels = ['positive', 'negative']
    plt.pie(x=data, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.show()
