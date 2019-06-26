
#This file classifies mushrooms as poison or edible using a random forest classifer
#Author: Sara Rabon
#Source: Data received from the follow Kaggle competition:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier #random forest classifier

import os
print(os.listdir("../input"))



#This program uses a training.csv file and test.csv file to classify mushrooms as poisonous
#or edible. It uses a random forest classifier from scikit learn to accomplish this.
def main():
    
    #Get data from training.csv and test.csv
    df = pd.read_csv("../input/train.csv", header=0)
    df_test = pd.read_csv("../input/test.csv", header=0)
    test_id = df_test["Id"]
    df_test = df_test.drop(columns = "Id")
    y = df.iloc[:,0]
    
    #combine into one data set
    combo = pd.concat([df, df_test], sort=False)
    
    #change from categorical to numerical values
    combo = pd.get_dummies(combo)
    
    #split back into training and test sets
    df = combo.head(len(df))
    df_test = combo.tail(len(df_test))
    x = df.iloc[:,3:]
    x_test = df_test.iloc[:,3:]
    
    #Fit random forest with training data and predict on test data
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(x,y)
    y_pred = classifier.predict(x_test)

    #combine into csv w/ lables from test and output new csv
    output = pd.DataFrame( y_pred, columns = ["class"])
    output = pd.concat([test_id, output], axis = 1)
    output.to_csv("csv_to_submit.csv", index = False)
    
if __name__ == "__main__":
    main()
