import json
from load_articles import read_articles
from value_keywords import read_valuation
import nltk


def write_keywords(keywords):
    with open('keywords.json', 'w') as fp:
        json.dump(keywords, fp)


def read_keywords():
    with open('keywords.json') as f:
        return json.load(f)


if __name__ == '__main__':
    valuation = read_valuation()
    # nltk.download('stopwords') only run this if 'stopwords' has not already been downloaded to your environment
    stopwords = nltk.corpus.stopwords.words('english')
    correlation = {k: abs(v) for k, v in valuation.items()}
    correlation = {k: v for k, v in sorted(correlation.items(), key=lambda item: item[1], reverse=True)}
    correlation = {k: v for k, v in correlation.items() if k not in stopwords and k not in ['', '—']}

    remove = ['\n', '…', '\xa0']
    keywords = [k for k in correlation.keys() if all(c not in k for c in remove)]
    write_keywords(keywords)
