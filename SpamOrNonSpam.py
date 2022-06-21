import os
import io
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)
            inBody = True
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = False
            f.close()
            message = '\n'.join(lines)
            yield path, message

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)
    return DataFrame(rows, index=index)


data = DataFrame({'message': [], 'class': []})
# read train data
data = data.append(dataFrameFromDirectory('data/spam-train', 'spam'))
data = data.append(dataFrameFromDirectory('data/nonspam-train', 'nonspam'))


vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

# train naive bayes classifier
classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)

testdata = DataFrame({'message': [], 'class': []})
testdata = testdata.append(dataFrameFromDirectory('data/nonspam-test', 'test'))  
examples = testdata['message'].values
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
#print(predictions)
numOfTrue = 0
numOfFalse = 0
for myclass in predictions:
    if myclass == 'nonspam':
        numOfTrue += 1
    if myclass == 'spam':
        numOfFalse += 1

print(numOfTrue)
print(numOfFalse)
# prediction
testdata = DataFrame({'message': [], 'class': []})
testdata = testdata.append(dataFrameFromDirectory('data/spam-test', 'test'))  
examples = testdata['message'].values
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
# print(predictions)
numOfTrue = 0
numOfFalse = 0
for myclass in predictions:
    if myclass == 'spam':
        numOfTrue += 1
    if myclass == 'nonspam':
        numOfFalse += 1

print(numOfTrue)
print(numOfFalse)

