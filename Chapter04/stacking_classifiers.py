# --- SECTION 1 ---
# Libraries
import numpy as np

from sklearn.model_selection import KFold
from copy import deepcopy


class StackingRegressor():

    # --- SECTION 2 ---
    # The constructor
    def __init__(self, learners):
        # Create a list of sizes for each stacking level
        # And a list of deep copied learners
        self.level_sizes = []
        self.learners = []
        for learning_level in learners:

            self.level_sizes.append(len(learning_level))
            level_learners = []
            for learner in learning_level:
                level_learners.append(deepcopy(learner))
            self.learners.append(level_learners)



    # --- SECTION 3 ---
    # The fit function. Creates training meta data for every level and trains
    # each level on the previous level's meta data
    def fit(self, x, y):
        # Create a list of training meta data, one for each stacking level
        # and another one for the targets. For the first level, the actual data
        # is used.
        meta_data = [x]
        meta_targets = [y]
        for i in range(len(self.learners)):
            level_size = self.level_sizes[i]

            # Create the meta data and target variables for this level
            data_z = np.zeros((level_size, len(x)))
            target_z = np.zeros(len(x))

            train_x = meta_data[i]
            train_y = meta_targets[i]

            # Create the cross-validation folds
            KF = KFold(n_splits=5)
            meta_index = 0
            for train_indices, test_indices in KF.split(x):
                # Train each learner on the K-1 folds and create
                # meta data for the Kth fold
                for j in range(len(self.learners[i])):

                    learner = self.learners[i][j]
                    learner.fit(train_x[train_indices], train_y[train_indices])
                    predictions = learner.predict(train_x[test_indices])

                    data_z[j][meta_index:meta_index+len(test_indices)] = predictions

                target_z[meta_index:meta_index+len(test_indices)] = train_y[test_indices]
                meta_index += len(test_indices)

            # Add the data and targets to the meta data lists
            data_z = data_z.transpose()
            meta_data.append(data_z)
            meta_targets.append(target_z)


            # Train the learner on the whole previous meta data
            for learner in self.learners[i]:
                    learner.fit(train_x, train_y)






    # --- SECTION 4 ---
    # The predict function. Creates meta data for the test data and returns
    # all of them. The actual predictions can be accessed with meta_data[-1]
    def predict(self, x):

        # Create a list of training meta data, one for each stacking level
        meta_data = [x]
        for i in range(len(self.learners)):
            level_size = self.level_sizes[i]

            data_z = np.zeros((level_size, len(x)))

            test_x = meta_data[i]

            # Create the cross-validation folds
            KF = KFold(n_splits=5)
            for train_indices, test_indices in KF.split(x):
                # Train each learner on the K-1 folds and create
                # meta data for the Kth fold
                for j in range(len(self.learners[i])):

                    learner = self.learners[i][j]
                    predictions = learner.predict(test_x)
                    data_z[j] = predictions



            # Add the data and targets to the meta data lists
            data_z = data_z.transpose()
            meta_data.append(data_z)

        # Return the meta_data the final layer's prediction can be accessed
        # With meta_data[-1]
        return meta_data



# --- SECTION 5 ---
# Use the classifier
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics
diabetes = load_diabetes()

train_x, train_y = diabetes.data[:400], diabetes.target[:400]
test_x, test_y = diabetes.data[400:], diabetes.target[400:]

base_learners = []

knn = KNeighborsRegressor(n_neighbors=5)
base_learners.append(knn)

dtr = DecisionTreeRegressor(max_depth=4, random_state=123456)
base_learners.append(dtr)

ridge = Ridge()
base_learners.append(ridge)

meta_learner = LinearRegression()

# Instantiate the stacking regressor
sc = StackingRegressor([[knn,dtr,ridge],[meta_learner]])

# Fit and predict
sc.fit(train_x, train_y)
meta_data = sc.predict(test_x)

# Evaluate base learners and meta learner
base_errors = []
base_r2 = []
for i in range(len(base_learners)):
    learner = base_learners[i]

    predictions = meta_data[1][:,i]

    err = metrics.mean_squared_error(test_y, predictions)
    r2 = metrics.r2_score(test_y, predictions)

    base_errors.append(err)
    base_r2.append(r2)

err = metrics.mean_squared_error(test_y, meta_data[-1])
r2 = metrics.r2_score(test_y, meta_data[-1])

# Print the results
print('ERROR  R2  Name')
print('-'*20)
for i in range(len(base_learners)):
    learner = base_learners[i]

    print(f'{base_errors[i]:.1f} {base_r2[i]:.2f} {learner.__class__.__name__}')
print(f'{err:.1f} {r2:.2f} Ensemble')
