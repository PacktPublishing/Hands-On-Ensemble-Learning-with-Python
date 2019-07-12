# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import time

start = time.time()

digits = load_digits()

train_size = 1500
train_x, train_y = digits.data[:train_size], digits.target[:train_size]
test_x, test_y = digits.data[train_size:], digits.target[train_size:]

# --- SECTION 2 ---
# Create our bootstrap samples and train the classifiers

ensemble_size = 10
base_learners = []

for _ in range(ensemble_size):
    # We sample indices in order to access features and targets
    bootstrap_sample_indices = np.random.randint(0, train_size, size=train_size)
    bootstrap_x = train_x[bootstrap_sample_indices]
    bootstrap_y = train_y[bootstrap_sample_indices]
    dtree = DecisionTreeClassifier()
    dtree.fit(bootstrap_x, bootstrap_y)
    base_learners.append(dtree)

# --- SECTION 3 ---
# Predict with the base learners and evaluate them
base_predictions = []
base_accuracy = []
for learner in base_learners:
    predictions = learner.predict(test_x)
    base_predictions.append(predictions)
    acc = metrics.accuracy_score(test_y, predictions)
    base_accuracy.append(acc)

# --- SECTION 4 ---
# Combine the base learners' predictions

ensemble_predictions = []
# Find the most voted class for each test instance
for i in range(len(test_y)):
    # Count the votes for each class
    counts = [0 for _ in range(10)]
    for learner_predictions in base_predictions:
        counts[learner_predictions[i]] = counts[learner_predictions[i]]+1

    # Find the class with most votes
    final = np.argmax(counts)
    # Add the class to the final predictions
    ensemble_predictions.append(final)

ensemble_acc = metrics.accuracy_score(test_y, ensemble_predictions)

end = time.time()


# --- SECTION 5 ---
# Print the accuracies
print('Base Learners:')
print('-'*30)
for index, acc in enumerate(sorted(base_accuracy)):
    print(f'Learner {index+1}: %.2f' % acc)
print('-'*30)
print('Bagging: %.2f' % ensemble_acc)

print('Total time: %.2f' % (end - start))
