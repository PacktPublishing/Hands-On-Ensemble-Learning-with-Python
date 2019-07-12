import json
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from tweepy import OAuthHandler, Stream, StreamListener

# Please fill your API keys as strings
consumer_key=""
consumer_secret=""

access_token=""
access_token_secret=""



# Load the data
data = pd.read_csv('sent140_preprocessed.csv')
data = data.dropna()
# Replicate our voting classifier for 30.000 features and 1-3 n-grams
train_size = 10000

tf = TfidfVectorizer(max_features=30000, ngram_range=(1, 3),
                         stop_words='english')
tf.fit(data.text)
transformed = tf.transform(data.text)

x_data = transformed[:train_size].toarray()
y_data = data.polarity[:train_size].values

voting = VotingClassifier([('LR', LogisticRegression()),
                                   ('NB', MultinomialNB()),
                                   ('Ridge', RidgeClassifier())])

voting.fit(x_data, y_data)


# Define the streaming classifier
class StreamClassifier(StreamListener):

    def __init__(self, classifier, vectorizer, api=None):
        super().__init__(api)
        self.clf = classifier
        self.vec = vectorizer

    # What to do when a tweet arrives
    def on_data(self, data):
        # Create a json object
        json_format = json.loads(data)
        # Get the tweet's text
        text = json_format['text']

        features = self.vec.transform([text]).toarray()
        print(text, self.clf.predict(features))
        return True

    # If an error occurs, print the status
    def on_error(self, status):
        print(status)

# Create the classifier and authentication handlers
classifier = StreamClassifier(classifier=voting, vectorizer=tf)
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Listen for specific hashtags
stream = Stream(auth, classifier)
stream.filter(track=['basketball'])