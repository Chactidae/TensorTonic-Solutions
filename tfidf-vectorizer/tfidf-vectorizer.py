import numpy as np
from collections import Counter
import math


def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    tokenized = [doc.split() for doc in documents]

    vocab = sorted(set(word for doc in tokenized for word in doc))

    vocab = {word: i for i, word in enumerate(vocab)}

    N = len(documents)

    df = np.zeros(len(vocab))

    for doc in tokenized:
        for word in set(doc):
            df[vocab[word]] += 1

    idf = np.log(N / df)

    tfdf = np.zeros((N, len(vocab)))

    for i, doc in enumerate(tokenized):
        counts = Counter(doc)

        for word, tf in counts.items():
            j = vocab[word]
            tf = tf / len(doc)
            tfdf[i, j] = tf * idf[j]

    return tfdf, vocab