import json
from load_articles import read_articles

dictionary = {}
net_words = 0


# think about how keywords are set to values
# format {'potato': 4, 'oil': 1}
def insert_dictionary(str):
    global net_words
    words = str.split()
    for word in words:
        net_words += 1
        if word in dictionary:
            dictionary[word] += 1
        else:
            dictionary[word] = 1


def get_ratios(dictionary_1):
    for key in dictionary_1.keys():
        dictionary_1[key] = dictionary_1[key] / net_words
    return dictionary_1


def write_dictionary():
    with open('dictionary.json', 'w') as fp:
        json.dump(dictionary, fp)


def read_dictionary():
    with open('dictionary.json') as f:
        return json.load(f)


if __name__ == '__main__':
    articles = read_articles()
    articles_length = len(articles)
    # divineRatio = (instance of word)/networds
    # after reading dictionary before writing it,
    # writing it is below -- have to divide all the values of dictionary by the articles
    for article in articles:
        insert_dictionary(article["title"])
        insert_dictionary(article["description"])

    # convert to ratios...
    dictionary = get_ratios(dictionary)
    write_dictionary()
    print(len(dictionary))
