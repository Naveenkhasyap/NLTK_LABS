from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "hello there Mr. Smith, how are you doing today? The weather is great today and you should not eat cardboard"

#print(sent_tokenize(example_text))

for i in word_tokenize(example_text):
    print(i)
