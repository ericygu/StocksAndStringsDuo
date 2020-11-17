import json
from load_articles import read_articles, write_articles
from form_dictionary import read_dictionary

def parse_keywords(articles, dictionary):

    def get_keywords(text, dictionary):
        word_list = text.split(' ')
        word_ratio = {word: text.count(word) / len(word_list)
                      for word in list(set(word_list))}
        keywords = []
        for word in word_ratio:
            if word in dictionary and word_ratio[word] < dictionary[word]:
                continue
            keywords.append(word)
        return keywords

    for article in articles:
        article['keywords'] = get_keywords(
            article['title'] + ' ' + article['description'], dictionary)
    return articles


if __name__ == '__main__':
    dictionary = read_dictionary()
    articles = read_articles()
    articles = parse_keywords(articles, dictionary)
    write_articles(articles)
