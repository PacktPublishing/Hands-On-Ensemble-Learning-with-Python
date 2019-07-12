import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
# Load the data
data = pd.read_csv('sent140_preprocessed.csv')
data = data.dropna()




# Set the train and test sizes
train_size = 10000
test_start = 10000
test_end = 100000



def check_features_ngrams(features, n_grams, classifiers):

    print(features, n_grams)

    # Create the IDF feature extractor
    tf = TfidfVectorizer(max_features=features, ngram_range=n_grams,
                         stop_words='english')

    # Create the IDF features
    tf.fit(data.text)
    transformed = tf.transform(data.text)

    np.random.seed(123456)

    def check_classifier(name, classifier):
        print('--'+name+'--')

        # Train the classifier
        x_data = transformed[:train_size].toarray()
        y_data = data.polarity[:train_size].values

        classifier.fit(x_data, y_data)
        i_s = metrics.accuracy_score(y_data, classifier.predict(x_data))

        # Evaluate on the test set
        x_data = transformed[test_start:test_end].toarray()
        y_data = data.polarity[test_start:test_end].values
        oos = metrics.accuracy_score(y_data, classifier.predict(x_data))

        # Expor the results
        with open("outs.txt","a") as f:
            f.write(str(features)+',')
            f.write(str(n_grams[-1])+',')
            f.write(name+',')
            f.write('%.4f'%i_s+',')
            f.write('%.4f'%oos+'\n')

    for name, classifier in classifiers:
        check_classifier(name, classifier)


# Create csv header
with open("outs.txt","a") as f:
    f.write('features,ngram_range,classifier,train_acc,test_acc')

# Test all features and n-grams combinations
for features in [500, 1000, 5000, 10000, 20000, 30000]:
    for n_grams in [(1, 1), (1, 2), (1, 3)]:

        # Create the ensemble
        voting = VotingClassifier([('LR', LogisticRegression()),
                                   ('NB', MultinomialNB()),
                                   ('Ridge', RidgeClassifier())])

        # Create the named classifiers
        classifiers = [('LR', LogisticRegression()),
                       ('NB', MultinomialNB()),
                       ('Ridge', RidgeClassifier()),
                       ('Voting', voting)]

        # Evaluate them
        check_features_ngrams(features, n_grams, classifiers)


