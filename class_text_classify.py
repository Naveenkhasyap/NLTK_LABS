import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_review.words(fileid),catrgory)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category))]

random.shuffle(documents)


#print(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())


all_words= nltk.FreqDist(all_words)
print(all_words.most_common(15))
#print(all_words["stupid"])
             
