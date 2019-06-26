import pandas as pd
from random import randint

from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def wordToNumber(word):
    if word == 'A':
        number = randint(81, 100)
    elif word == 'AB':
        number = randint(71, 80)
    elif word == 'B':
        number = randint(66, 70)
    elif word == 'BC':
        number = randint(61, 65)
    elif word == 'C':
        number = randint(56, 60)
    elif word == 'D':
        number = randint(41, 55)
    elif word == 'E':
        number = randint(0, 40)
    else:
        number = randint(60, 100)
    return number

def main():
    data = pd.read_csv("data.csv")

    for idx, row in data.iterrows():
        data.at[idx, 'ML'] = wordToNumber(row['ML'])
        data.at[idx, 'DW'] = wordToNumber(row['DW'])
    
    mldw_data = data.loc[:, 'ML':].values

    clf = IsolationForest( behaviour = 'new', max_samples=28, random_state = 1, contamination= 'auto')
    preds = clf.fit_predict(mldw_data)
    
    # print(data['NAMA'][1])

    i = 0
    outliers_idx = []
    for predict in preds:
        i+=1
        if predict == -1:
            outliers_idx.append(i)

    for i in range(0, len(preds)):
        j = i + 1
        if j in outliers_idx:
            plt.scatter(mldw_data[i][0], mldw_data[i][1], c="red")
            plt.annotate(data['NAMA'][i], (mldw_data[i][0], mldw_data[i][1]))
        else:
            plt.scatter(mldw_data[i][0], mldw_data[i][1], c="black")
    
    plt.show()
    

if __name__ == "__main__":
    main()