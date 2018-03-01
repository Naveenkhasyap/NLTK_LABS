#lemmatizing similar to stemming
import nltk

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))


print(lemmatizer.lemmatize("better",pos="a")) # a is adjective default it will be n noun
print(lemmatizer.lemmatize("best",pos="a"))
print(lemmatizer.lemmatize("run"))









