import json
import re
from load_articles import read_articles, write_articles
from form_dictionary import read_dictionary
import datetime
import requests

def main():
   exec(open("./load_articles.py").read())
   exec(open("./form_dictionary.py").read())
   exec(open("./form_keywords.py").read())
   exec(open("./stock_parse.py").read())
   exec(open("./value_keywords.py").read())

if __name__ == '__main__':
    main()