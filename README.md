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

## Donate us
If you like our project and it's useful, feel free to donate us.

Bitcoin(BTC): bc1qs2fatdfdvc5jyq4a0f5t7plmy8sxmyk08tq5e5

Ethereum(ETH): 0x5847D46Bfed82a475ef4187cfBD55EF412C05093

Tether(USDT-TRC20): TAmbZwJXDZ8bo2hjGXtNkTSEYi8dt2Xww8

XRP(XRP): rqTpCtGtBEhcPjZLXfNTv3JbCdkRKGLCF

Dogecoin(DOGE): DGZYMS6nnT3cBYwDtSD7VVubr1dSfykURC

TRON(TRX): TAmbZwJXDZ8bo2hjGXtNkTSEYi8dt2Xww8

BitTorrent-New(BTT-BEP20): 0x5847D46Bfed82a475ef4187cfBD55EF412C05093

Decentraland(MANA-ERC20): 0x5847D46Bfed82a475ef4187cfBD55EF412C05093

