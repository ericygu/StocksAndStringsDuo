from load_articles import read_articles
import json


def write_valuation(valuation_dictionary):
    with open('valuation.json', 'w') as fp:
        json.dump(valuation_dictionary, fp)


def read_valuation():
    with open('valuation.json') as f:
        return json.load(f)


if __name__ == '__main__':
    articles = read_articles()

    valuation_dictionary = {}
    for article in articles:
        valuation = article['delta']

        for keyword in article['keywords']:
            if keyword not in valuation_dictionary:
                valuation_dictionary[keyword] = 0
            valuation_dictionary[keyword] += valuation
    write_valuation(valuation_dictionary)
