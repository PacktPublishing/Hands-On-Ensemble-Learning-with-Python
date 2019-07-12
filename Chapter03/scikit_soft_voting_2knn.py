#  --- SECTION 1 ---
# Import the required libraries
from sklearn import datasets, naive_bayes, svm, neighbors
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
# Load the dataset
breast_cancer = datasets.load_breast_cancer()
x, y = breast_cancer.data, breast_cancer.target

# Split the train and test samples
test_samples = 100
x_train, y_train = x[:-test_samples], y[:-test_samples]
x_test, y_test = x[-test_samples:], y[-test_samples:]

#  --- SECTION 2 ---
# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = naive_bayes.GaussianNB()
learner_3 = neighbors.KNeighborsClassifier(n_neighbors=50)

#  --- SECTION 3 ---
# Instantiate the voting classifier
voting = VotingClassifier([('5NN', learner_1),
                           ('NB', learner_2),
                           ('50NN', learner_3)],
                            voting='soft')




#  --- SECTION 4 ---
# Fit classifier with the training data
voting.fit(x_train, y_train)
learner_1.fit(x_train, y_train)
learner_2.fit(x_train, y_train)
learner_3.fit(x_train, y_train)

#  --- SECTION 5 ---
# Predict the most probable class
hard_predictions = voting.predict(x_test)

#  --- SECTION 6 ---
# Get the base learner predictions
predictions_1 = learner_1.predict(x_test)
predictions_2 = learner_2.predict(x_test)
predictions_3 = learner_3.predict(x_test)

#  --- SECTION 7 ---
# Accuracies of base learners
print('L1:', accuracy_score(y_test, predictions_1))
print('L2:', accuracy_score(y_test, predictions_2))
print('L3:', accuracy_score(y_test, predictions_3))
# Accuracy of hard voting
print('-'*30)
print('Hard Voting:', accuracy_score(y_test, hard_predictions))

#  --- SECTION 1 ---
# Import the required libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('seaborn-paper')


#  --- SECTION 2 ---
# Get the wrongly predicted instances
# and the predicted probabilities for the whole test set
errors = y_test-hard_predictions

probabilities_1 = learner_1.predict_proba(x_test)
probabilities_2 = learner_2.predict_proba(x_test)
probabilities_3 = learner_3.predict_proba(x_test)


#  --- SECTION 2 ---
# Store the predicted probability for
# each wrongly predicted instance, for each base learner
# as well as the average predicted probability
#
x=[]
y_1=[]
y_2=[]
y_3=[]
y_avg=[]

for i in range(len(errors)):
    if not errors[i] == 0:
        x.append(i)
        y_1.append(probabilities_1[i][0])
        y_2.append(probabilities_2[i][0])
        y_3.append(probabilities_3[i][0])
        y_avg.append((probabilities_1[i][0]+probabilities_2[i][0]+probabilities_3[i][0])/3)

#  --- SECTION 3 ---
# Plot the predicted probaiblity of each base learner as
# a bar and the average probability as an X
plt.bar(x, y_1, 3,   label='5NN')
plt.bar(x, y_2, 2,   label='NB')
plt.bar(x, y_3, 1,   label='50NN')
plt.scatter(x, y_avg, marker='x', c='k', s=150, label='Average Positive', zorder=10)

y = [0.5 for x in range(len(errors))]
plt.plot(y, c='k', linestyle='--')

plt.title('Positive Probability')
plt.xlabel('Test sample')
plt.ylabel('probability')
plt.legend()






