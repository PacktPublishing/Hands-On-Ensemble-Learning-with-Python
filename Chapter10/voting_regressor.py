import numpy as np
from copy import deepcopy

class VotingRegressor():

    # Accepts a list of (name, classifier) tuples
    def __init__(self, base_learners):
        self.base_learners = {}
        for name, learner in base_learners:
            self.base_learners[name] = deepcopy(learner)


    # Fits each individual base learner
    def fit(self, x_data, y_data):
        for name in self.base_learners:
            learner = self.base_learners[name]
            learner.fit(x_data, y_data)

    # Generates the predictions
    def predict(self, x_data):

        # Create the predictions matrix
        predictions = np.zeros((len(x_data), len(self.base_learners)))

        names = list(self.base_learners.keys())

        # For each base learner
        for i in range(len(self.base_learners)):
            name = names[i]
            learner = self.base_learners[name]

            # Store the predictions in a column
            preds = learner.predict(x_data)
            predictions[:,i] = preds

        # Take the row-average
        predictions = np.mean(predictions, axis=1)
        return predictions