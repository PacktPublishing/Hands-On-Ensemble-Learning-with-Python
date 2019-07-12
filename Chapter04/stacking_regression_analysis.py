import matplotlib.pyplot as plt

knn_d = test_meta_data[:,0]-test_y
dtr_d = test_meta_data[:,1]-test_y
ridge_d = test_meta_data[:,2]-test_y
meta_d =  ensemble_predictions-test_y

plt.plot(knn_d, label='KNN')
plt.plot(dtr_d, label='DTree')
plt.plot(ridge_d, label='Ridge')
plt.plot(meta_d, label='Ensemble')
plt.legend()