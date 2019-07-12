# --- SECTION 1 ---
# Libraries
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import time


from concurrent.futures import ProcessPoolExecutor

# --- SECTION 2 ---
# Define required functions
train_size = 1500


def create_learner(train_x, train_y):
    # We sample indices in order to access features and targets
    bootstrap_sample_indices = np.random.randint(0, train_size, size=train_size)
    bootstrap_x = train_x[bootstrap_sample_indices]
    bootstrap_y = train_y[bootstrap_sample_indices]
    dtree = DecisionTreeClassifier()
    dtree.fit(bootstrap_x, bootstrap_y)
    return dtree


def predict(learner, test_x):
    return learner.predict(test_x)


# --- SECTION 3 ---
# Protect our main
if __name__ == '__main__':

    start = time.time()
    digits = load_digits()

    train_x, train_y = digits.data[:train_size], digits.target[:train_size]
    test_x, test_y = digits.data[train_size:], digits.target[train_size:]

    ensemble_size = 1000
    base_learners = []

    # --- SECTION 4 ---
    # Create the base learners
    with ProcessPoolExecutor() as executor:
        futures = []
        for _ in range(ensemble_size):
            future = executor.submit(create_learner, train_x, train_y)
            futures.append(future)

        for future in futures:
            base_learners.append(future.result())

    # --- SECTION 5 ---
    # Predict with the base learners and evaluate them
    base_predictions = []
    base_accuracy = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for learner in base_learners:
            future = executor.submit(predict, learner, test_x)
            futures.append(future)

        for future in futures:
            predictions = future.result()
            base_predictions.append(predictions)
            acc = metrics.accuracy_score(test_y, predictions)
            base_accuracy.append(acc)

    # --- SECTION 6 ---
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

    # --- SECTION 7 ---
    # Print the accuracies
    print('Base Learners:')
    print('-'*30)
    for index, acc in enumerate(sorted(base_accuracy)):
        print(f'Learner {index+1}: %.2f' % acc)
    print('-'*30)
    print('Bagging: %.2f' % ensemble_acc)
    print('Total time: %.2f' % (end - start))
