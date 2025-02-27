#Import Required Libraries

import pandas as pd
import requests
import io
from zipfile import ZipFile
import pickle


# Download the SMS Spam Collection dataset from the UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
r = requests.get(url)
z = ZipFile(io.BytesIO(r.content))

# Read the dataset file into a Pandas DataFrame
filename = 'SMSSpamCollection'
with z.open(filename) as file:
    df = pd.read_csv(file, sep='\t', header=None, names=['v1', 'v2'])

# Convert labels to binary values
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2, random_state=42)


# Build a text classification pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB())
])

# Train the model on the training set
text_clf.fit(X_train, y_train)

# Evaluate the model on the testing set
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = text_clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification report:\n', classification_report(y_test, y_pred))

# Save the trained model as a pickle file Most imp
with open('spam_classifier.pkl', 'wb') as file:
    pickle.dump(text_clf, file)