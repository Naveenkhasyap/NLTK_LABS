# Natural language processing invvloves many steps in making system
# understand 
#


import nltk

text = " mary had a little lamb. Her fience was white snow"

from nltk.tokenize import word_tokenize,sent_tokenize
#sentence tokenizing 
sents = sent_tokenize(text)
#print(sents)

#word tokenizing 
words= word_tokenize(text)
#print(words)

#stop word removal
from nltk.corpus import stopwords
from string import punctuation
customStopWords = set(stopwords.words('english')+list(punctuation))

wordsWOStopwords = [word for word in word_tokenize(text) if word not in customStopWords]
#print(wordsWOStopwords)

#n grams 
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(wordsWOStopwords)

sorted(finder.ngram_fd.items())
#print(sorted(finder.ngram_fd.items()))

#stemming and tagging parts of speech
text2 = "mary closed on closing night when she was about to close" 
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
stemmedWords = [st.stem(word) for word in word_tokenize(text2)]
#print(stemmedWords)

#pos tagging
#print(nltk.pos_tag(word_tokenize(text2)))

#word sense disambiguation word net is like a dictionary
from nltk.corpus import wordnet as wn
#for ss in wn.synsets('bass'):
    #print(ss,ss.definition())

#word sense disambiguation
from nltk.wsd import lesk
sensel = lesk(word_tokenize("sing in a lower tone,along with the bass"),'bass')
print(sensel,sensel.definition)

sense2 = lesk(word_tokenize("This sea bass was really had to catch"),'bass')
print(sense2,sense2.definition())
