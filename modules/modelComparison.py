"""
@author: Shravankumar Hiregoudar
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def resultCompare(X,targets):
    """
        :resultCompare: Compare the results of different (RF,SVC,LogisticR) with their defalut parameters to understand the precision,
                        recall, F1 score on validation set and Training accuracy, Testing Accuracy and validation set accuracy of the models.
        :param X: Dataframe containing training features
        :type X: DataFrame

        :param targets: Dataframe of training targets(Stability vectors)
        :type targets: DataFrame

        :returns: DataFrame containing model information with all the result metrics 
    """

    # Make train,test and validation split
    X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.3)
    X_train2, X_dev, y_train2, y_dev = train_test_split(X_train, y_train, test_size=0.1)

    for i, col in enumerate(y_train2.columns.tolist(), 1):
        y_train2.loc[:, col] *= i
        yTrain = y_train2.sum(axis=1)

    for i, col in enumerate(y_dev.columns.tolist(), 1):
        y_dev.loc[:, col] *= i
        yDev = y_dev.sum(axis=1)

    for i, col in enumerate(y_test.columns.tolist(), 1):
        y_test.loc[:, col] *= i
        yTest = y_test.sum(axis=1)

    # Define the models. You can add models to compare
    lg = LogisticRegression()
    svc = SVC()
    rf = RandomForestClassifier()
    
    models = [lg,svc,rf]
    trainAccuracy,testAccuracy,devAccuracy = [],[],[]
    
    # For each of the models
    for model in models:
        # Fit on the training data
        model.fit(X_train2, yTrain)

        # Predict on the dev set to make comparision between models
        predict  = model.predict(X_dev)

        actual = np.array(yDev).flatten()
        predicted = predict.flatten()

        # Get the accuracy
        trainAccuracy.append(accuracy_score(yTrain, model.predict(X_train2) ))
        devAccuracy.append(accuracy_score(yDev,model.predict(X_dev)))
        testAccuracy.append(accuracy_score(yTest,model.predict(X_test)))

        print("\n\n\n RESULTS OF ",type(model).__name__)
        getScores(actual, predicted)

    accuracyDF = pd.DataFrame(data={'Model name': models, 'Train Accuracy': trainAccuracy, 'Test Accuracy': testAccuracy, 'Dev Accuracy': devAccuracy})
    return(accuracyDF)

def getScores(y_test, y_pred):
    """
        :getScores: Function to get classification_report along with micro,macro and weighted precision, recall and F1 score

        :param y_test: Actual values
        :type y_test: np.array

        :param y_pred: Predicted values
        :type y_pred: np.array

        :returns: classification_report along with micro,macro and weighted precision, recall and F1 score
    """


    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix\n')
    print(confusion)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

    from sklearn.metrics import classification_report
    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred))