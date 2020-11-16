import json
import re

API_KEY = '66c2e12a4d234e33b3df8d3057ea4a5a'


def get_articles(source='bbc-news'):
    from newsapi import NewsApiClient
    newsapi = NewsApiClient(api_key=API_KEY)

    def get_page(n):
        return newsapi.get_everything(
            q='facebook',
            sources=source,
            from_param='2020-11-15',
            language='en',
            sort_by='relevancy',
            page=n
        )

    def parse_article(article):
        p = re.compile('[\^\'!\.?:,\-\"\\\/]+')
        return {
            'title': p.sub("", article['title'].lower()),
            # 'content': p.sub("", article['content'].lower()),
            'description': p.sub("", article['description'].lower()),
            'date_published': article['publishedAt']
        }

    page_1 = get_page(1)
    hits = page_1['totalResults']
    articles = []
    cur_page = 1

    while len(articles) < hits:
        print(str(len(articles)) + '/' + str(hits))
        try:
            page = get_page(cur_page)
        except:
            break
        articles += [parse_article(art) for art in page['articles']]
        cur_page += 1

    # print(articles)
    return articles


def write_articles(articles):
    with open('articles.json', 'w') as fp:
        json.dump(articles, fp)


def read_articles():
    with open('articles.json') as f:
        return json.load(f)


if __name__ == '__main__':
    sources = ['bbc-news', 'abc-news', 'australian-financial-review',
               'business-insider', 'business-insider-uk', 'cnbc', 'financial-post',
               'fortune', 'the-wall-street-journal']
    articles = []
    for source in sources:
        articles += get_articles(source)
    write_articles(articles)
