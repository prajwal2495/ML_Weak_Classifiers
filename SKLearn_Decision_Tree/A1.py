from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

if __name__ == "__main__":
    #  Load training data
    data_train = pd.read_csv("../data/audiology_train.csv")
    # Separate independent variables and dependent variables
    independent = range(69)
    X = data_train[independent]
    y = data_train[70]
    # Train model
    clf = GaussianNB()
    clf.fit(X,y)
    # Load testing data
    data_test = pd.read_csv("../data/audiology_test.csv")
    X_test = data_test[independent]
    # Predict
    predictions = clf.predict(X_test)
    # Predict probabilities
    probs = clf.predict_proba(X_test)
    # Print results
    for i,pred in enumerate(predictions):
        print("%s\t%f" %(pred,max(probs[i])))
