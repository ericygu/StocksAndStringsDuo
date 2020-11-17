from load_articles import *
from form_dictionary import *
from form_keywords import*
from stock_parse import *
from value_keywords import *
    
def main():
   exec(open("./load_articles.py").read())
   exec(open("./form_dictionary.py").read())
   exec(open("./form_keywords.py").read())
   exec(open("./stock_parse.py").read())
   exec(open("./value_keywords.py").read())

if __name__ == '__main__':
    main()