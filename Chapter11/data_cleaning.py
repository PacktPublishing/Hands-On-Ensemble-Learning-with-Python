
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation

# Read the data and assign labels
labels = ['polarity', 'id', 'date', 'query', 'user', 'text']
data = pd.read_csv("sent140.csv", names=labels)

# Keep only text and polarity, change polarity to 0-1
data = data[['text', 'polarity']]
data.polarity.replace(4, 1, inplace=True)

# Create a list of stopwords
stops = stopwords.words("english")

# Add stop variants without single quotes
no_quotes = []
for word in stops:
    if "'" in word:
        no_quotes.append(re.sub(r'\'', '', word))
stops.extend(no_quotes)


def clean_string(string):
    # Remove HTML entities
    tmp = re.sub(r'\&\w*;', '', string)
    # Remove @user
    tmp = re.sub(r'@(\w+)', '', tmp)
    # Remove links
    tmp = re.sub(r'(http|https|ftp)://[a-zA-Z0-9\\./]+', '', tmp)
    # Lowercase
    tmp = tmp.lower()
    # Remove Hashtags
    tmp = re.sub(r'#(\w+)', '', tmp)
    # Remove repeating chars
    tmp = re.sub(r'(.)\1{1,}', r'\1\1', tmp)
    # Remove anything that is not letters
    tmp = re.sub("[^a-zA-Z]", " ", tmp)
    # Remove anything that is less than two characters
    tmp = re.sub(r'\b\w{1,2}\b', '', tmp)
    # Remove multiple spaces
    tmp = re.sub(r'\s\s+', ' ', tmp)
    return tmp



def preprocess(string):

    stemmer = PorterStemmer()
    # Remove any punctuation character
    removed_punc = ''.join([char for char in string if char not in punctuation])

    cleaned = []
    # Remove any stopword
    for word in removed_punc.split(' '):
        if word not in stops:
            cleaned.append(stemmer.stem(word.lower()))
    return ' '.join(cleaned)




# Shuffle
data = data.sample(frac=1).reset_index(drop=True)
# Clean
data.text = data.text.apply(clean_string)
# Pre-process
data.text = data.text.apply(preprocess)
# Save to CSV
data.to_csv('sent140_preprocessed.csv', index=False)