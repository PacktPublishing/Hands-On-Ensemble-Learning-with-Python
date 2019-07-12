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
plt.bar(x, y_1, 3,   label='KNN')
plt.bar(x, y_2, 2,   label='NB')
plt.bar(x, y_3, 1,   label='SVM')
plt.scatter(x, y_avg, marker='x', c='k', s=150, label='Average Positive', zorder=10)

y = [0.5 for x in range(len(errors))]
plt.plot(y, c='k', linestyle='--')

plt.title('Positive Probability')
plt.xlabel('Test sample')
plt.ylabel('probability')
plt.legend()
plt.show()