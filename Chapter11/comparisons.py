import matplotlib.pyplot as plt




res = pd.read_csv('outs_old.csv')

i = 1
for key, grp in res.groupby(['classifier']):
    plt.subplot(2, 2, i)
    i += 1
    plt.title(str(key))
    for key2, grp2 in grp.groupby(['ngram_range']):
        plt.plot(grp2.features.values, grp2['test_acc'].values, label=str(key2)+'-gram')
   # plt.xscale('log')
    plt.legend()
    plt.xlabel('features')
    plt.ylabel('accuracy')
    plt.xscale('log')