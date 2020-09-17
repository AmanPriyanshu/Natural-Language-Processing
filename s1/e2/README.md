# Bayes Rule for NLP:

## Bayes Rule:
Bayes' Theorem is a way of finding a probability when we know certain other probabilities. The formula is: P(A|B) = P(A) P(B|A)P(B).
It uses conditional probabilities.
Now, let us explore methods to apply this rule into NLP classification tasks.

`P(+|"happy") = P("happy"|+) x P(+)/p("happy")`

P(+|"happy") = probability of a tweet being positive provided it has ther word "happy" in it.

## Naive Bayes - Supervised Learning
Find out the number of times a word is repeated in positive tweets and number of times they are repeated in negative tweets as well.
Calculate the number of total words present in positivd tweets and number of words in negative tweets. Then calculate the probability of each word by,

P(word|+) = number_of_times_word_repeated_in_positive/Number_of_total_words_in_positive

Thereby, we get a ratio of P(w|+), i.e. probability of the tweet containing the word w when the tweet is positive.

## Naive Bayes Inference Condition Rule:
output = product( P(w|+)/P(w|-) )

In this case if output > 1, then it is simply a positive tweet.

## Laplacian Smoothing:

P(w|+) = freq(w, +)/N(+), however, this could go wrong since words which were never present in the negative corpus would give out a probability of 0, thereby, undermining the rest of the probabilities of that corpus.
To smooth this problem, we introduce Laplacian Smoothing.

`P(w|+) = ( freq(w, +) + 1 )/( N(+) + V )`,

where V is the vocabulary size of the corpus (both + and -).

Allows for easier and smoother calculation.

## Log Likelihood:

`ratio(w) = P(w|+)/P(w|-)

* positive words have ratio > 1
* negative words have ratio < 1
* neutral words have ratio = ~1

To make calculations faster, we can use logarithms,

`lambda(w) = log( P(w|+)/P(w|-) )` or `log( ratio )`

* positive words have lambda > 0
* negative words have lambda < 0
* neutral words have lambda = ~0

Check out the notebook for implementation