from urllib.request import urlopen
from bs4 import BeautifulSoup

articleURL = "https://www.washingtonpost.com/news/the-switch/wp/2016/10/18/the-pentagons-massive-new-telescope-is-designed-to-track-space-junk-and-watch-out-for-killer-asteroids/?utm_term=.19d9797a72d0"

page = urlopen(articleURL).read().decode('utf8','ignore')
soup = BeautifulSoup(page,'lxml')

#first article element 
#print(soup.find('article').text)

#text = ' '.join(map(lambda p:p.text,soup.find_all('article')))

#generic function to download article and process it
def getTextWaPo(url):
    page = urlopen(url).read().decode('utf8')
    soup = BeautifulSoup(page,"lxml")
    text = ' '.join(map(lambda p: p.text, soup.find_all('article')))
    return text.encode('ascii', errors='replace').replace("?"," ")
#text = text.encode('ascii','replace').replace('?',' ')
#print(text)
text = getTextWaPo(articleURL)
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from string import punctuation

sents = sent_tokenize(text)
#print(sents)

word_sents = sent_tokenize(text)
#print(word_sents)

from nltk.probability import FreqDist 
freq = FreqDist(word_sents)

from heapq import nlargest 
nlargest(10,freq,key=freq.get)
print(nlargest)

from collections import defaultdict
ranking = defaultdict(int)

for i,sent in enumerate(sents):
    for w in word_tokenize(sent.lower()):
        if w in freq:
            ranking[i] += freq[w]

print(ranking)
