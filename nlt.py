import nltk
nltk.download()
sentence = """At ten o'clock on friday morning sam didn't feel very good."""
tokens = nltk.word_tokenize(sentence)
print(tokens)

tagged = nltk.pos_tag(tokens)
tagged[0:6]
print(tagged)
