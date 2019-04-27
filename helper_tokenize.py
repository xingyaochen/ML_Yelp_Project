def removeStopWords(tokenized_sent):
    stop_words=set(stopwords.words("english"))
    filtered_sent=[]
    for w in tokenized_sent:
        if w not in stop_words:
            filtered_sent.append(w)
    return filtered_sent

def convertToStem(tokenized_sent):
    ps = PorterStemmer()
    for word in tokenized_sent:
        word=ps.stem(word)
    return tokenized_sent

def preprocess_Review(tokenized_sent):
    stop_words=set(stopwords.words("english"))
    ps = PorterStemmer()
    filtered_sent=[]
    for w in tokenized_sent:
        w=w.lower()
        if w not in stop_words:
            word=ps.stem(w)
            filtered_sent.append(word)
    return filtered_sent

def bigram(tokenized_sent):
    return list(nltk.bigrams(tokenized_sent))

def trigram(tokenized_sent):
    return list(nltk.trigrams(tokenized_sent))
