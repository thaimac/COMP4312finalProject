import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


def pkl_write(data, filename='data.pickle'):
    with open(filename, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def pkl_read(filename='data.pickle'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def save(model, filename):
    print("- Save the model...")
    pkl_write(model, filename + ".pickle")
    print("\t+ Done.")


def load(filename):
    print("- Load the model...")
    model = pkl_read(filename)
    print("\t+ Done.")
    return model


def evaluate(model, xtest, ytest):
    predictions = model.predict(xtest)
    errors = abs(predictions - ytest)
    mape = 100 * np.mean(errors/ytest)
    accuracy = 100-mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))


def tune(rfr, xtrain, ytrain, xtest, ytest):
    param_grid = [
    # try 16 (4×4) combinations of hyperparameters
    {'n_estimators': [3, 6, 10, 30], 'max_features': [2, 4, 6, 8]},
    #then try 20 (4×5) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 5, 7, 10], 'max_features': [2, 3, 4, 6, 7]},
    ]

    # train across 5 folds, that's a total of (16+20)*5=180 rounds of training!
    grid_search = GridSearchCV(rfr, param_grid, cv=5, return_train_score=True)
    grid_search.fit(xtrain, ytrain)
    print("\nBest parameters: ", grid_search.best_params_)
    print("Best estimator: ", grid_search.best_estimator_)

    best_rfr = grid_search.best_estimator_
    best_rfr.fit(xtrain, ytrain)

    #evaluate newly tuned model
    evaluate(best_rfr, xtest, ytest)

    #save tuned model
    save(best_rfr, "./data/tuned_RFR")


def train():
    trainset = pd.read_csv("./data/train2.csv", index_col=0)
    testset = pd.read_csv("./data/test2.csv", index_col=0)
    xtrain, ytrain = trainset.drop(columns=['median_house_value']), trainset['median_house_value']
    xtest, ytest = testset.drop(columns=["median_house_value"]), testset["median_house_value"]

    print("-----TRAINING THE BASELINE-----")
    rfr = RandomForestRegressor(n_estimators=10, random_state=42)

    #train model
    rfr.fit(xtrain, ytrain)

    #evaluate untuned model
    evaluate(rfr, xtest, ytest)

    #tune model
    tune(rfr, xtrain, ytrain, xtest, ytest)


def predict(sent, model_name):
    model = load(model_name)
    predicted_price = model.predict([sent]).tolist()[0]
    print("- Inference...")
    print("\t+ %s with" % (predicted_price))
    return predicted_price


if __name__ == '__main__':
    train()