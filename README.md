# Spam_detection

An Naive Bayes classifier for email spam detection

## How it works

The function "readFiles" read all data files from "data" directory.

The function "dataFrameFromDirectory" make a Dataframe for files.

We extract the words and the number of times they are repeated from the dataframe containing the emails' content with the help of the learn-Scikit library and
the CountVectorizer() method.

Below the array of counts is a two-dimensional matrix in which the rows of individual emails and columns are words, with the amount of repetition of words in each 
email written in rows.

```python
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)
```

Finally, we give the words repeated in each email to the Naive Bayes classifier to model based on the type of email that is spam or not:

```python
classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)
```
