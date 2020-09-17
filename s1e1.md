# Simple Logistic Regression For Vector Classification

## Logistic Regression - Supervised Learning:
Supervised Learning is a branch of machine learning which deals with data containing both inputs and their respective outputs.
That is both X and Y is given as input to the model and the model's aim is to map X to Y.

## Vocabulary:
Vocabulary is an integral part of understanding language. Without basic understanding of the vocabulary of language, an entity cannot decipher its meaning.
There are various methods by which machine learning models can handle vocabulary, let's look at a couple of them here,

### Sparse Representation of Vocabulary/Bag of Words Method:
Having given documents D[i], where, i belongs to [1, N] with a total of 4*N words, we can generate a vocabulary having only specific words, say N/2 words only. Therefore,
 when we begin encoding our documents we will use sparse representation, i.e. if a word falls into vocabulary it is represented as 1 otherwise it is represented as 0.

Drawbacks: heavily depends on vocabulary size.

### Positive and Negative Counts (Classification):
Count the frequncy of a particular word in positive tweets, similarly for negative tweets. We will get 2 columns, one filled with positive frequency another with negative frequency.

Representation of document d would be given as X[d] = [1, sum(Freq(w, 1)), sum(Freq(w, 0))]

Drawbacks: Does not consider sequence of words.

## Preprocessing:
Preprocessing is an integral part of Natural Language Processing.

### Stop Words and Punctuations:
* Stop Words: In computing, stop words are words which are filtered out before or after processing of natural language data.
* Punctuations: Punctuation is the use of spacing, conventional signs, and certain typographical devices as aids to the understanding and correct reading of written text, whether read silently or aloud.

### Stemming and lowercasing:
* stemming: transforming words to their base stem
* lowercasing: allows models to treat words the same irrespective of thier case.

## Logistic Regression:
A form of supervised machine learning, it uses the activation function sigmoid, which maps Z = w.x + b to [0, 1]. It is useful for binary classification which is eactly are problem statement right now.
Easy to implement for numeric data, which we have alreayd implemented using Positive Negative Count, we will run it for 50 epochs and check its accuracy and loss (binary_crossentropy), the Adam Optimizer has been used.

Check out the notebook for code