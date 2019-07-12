# --- SECTION 1 ---
# Libraries and data loading
from copy import deepcopy
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import numpy as np

bc = load_breast_cancer()

train_size = 400
train_x, train_y = bc.data[:train_size], bc.target[:train_size]
test_x, test_y = bc.data[train_size:], bc.target[train_size:]

np.random.seed(123456)

# --- SECTION 2 ---
# Create the ensemble
ensemble_size = 100
base_classifier = DecisionTreeClassifier(max_depth=1)

# Create the initial weights
data_weights = np.zeros(train_size) + 1/train_size
# Create a list of indices for the train set
indices = [x for x in range(train_size)]

base_learners = []
learners_errors = np.zeros(ensemble_size)
learners_weights = np.zeros(ensemble_size)

errs = []
# Create each base learner
for i in range(ensemble_size):
    weak_learner = deepcopy(base_classifier)

    # Choose the samples by sampling with replacement.
    # Each instance's probability is dictated by its weight.
    data_indices = np.random.choice(indices, train_size, p=data_weights)
    sample_x, sample_y = train_x[data_indices], train_y[data_indices]

    # Fit the weak learner and evaluate it
    weak_learner.fit(sample_x, sample_y)
    predictions = weak_learner.predict(train_x)

    errors = predictions != train_y
    corrects = predictions == train_y

    # Calculate the weighted errors
    weighted_errors = data_weights*errors


    # The base learner's error is the average of the weighted errors
    learner_error = np.mean(weighted_errors)
    learners_errors[i] = learner_error

    # The learner's weight
    learner_weight = np.log((1-learner_error)/learner_error)/2
    learners_weights[i] = learner_weight

    # Update the data weights
    data_weights[errors] = np.exp(data_weights[errors] * learner_weight)
    data_weights[corrects] = np.exp(-data_weights[corrects] * learner_weight)

    data_weights = data_weights/sum(data_weights)
    # Save the learner
    base_learners.append(weak_learner)



# --- SECTION 3 ---
# Evaluate the ensemble
ensemble_predictions = []
for learner, weight in zip(base_learners, learners_weights):
    # Calculate the weighted predictions
    prediction = learner.predict(test_x)
    ensemble_predictions.append(prediction*weight)

# The final prediction is the weighted mean of the individual predictions
ensemble_predictions = np.mean(ensemble_predictions, axis=0) >= 0.5

ensemble_acc = metrics.accuracy_score(test_y, ensemble_predictions)

# --- SECTION 4 ---
# Print the accuracy
print('Boosting: %.2f' % ensemble_acc)
