"""
@author: Shravankumar Hiregoudar
"""
from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.interpolate import make_interp_spline, BSpline
    
def RFmodel(X,targets,hyperParameters,plotCM = False,getOnlyModel = True):
    """
        :getRFResults: Random Forest model is built for the hyperparameter combinations and parameters to evaluate the models are calculated
        :param X: Train data
        :type X: DataFrame

        :param targets: Test data
        :type targets: DataFrame

        :param hyperParameters: Hyperparameter combinations
        :type hyperParameters: list

        :param plotCM: If True, Plot the confusion matrix
        :type plotCM: bool

        :param getOnlyModel: If True, Return the RF model and break the function
        :type getOnlyModel: bool

        :returns: Evaluation metrics such as F1, recall, accuracy etc 
    """
    times = []
    precision = []
    recall =[]
    f1 = []
    trainAccuracy = []
    devAccuracy = []
    testAccuracy = []
    
    # Make train,test and validation split
    X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.3)
    X_train2, X_dev, y_train2, y_dev = train_test_split(X_train, y_train, test_size=0.1)

    # For each combination of hyperparameters
    for param in hyperParameters:
        start = time.time()
        rf = RandomForestClassifier(max_features = param[0],     
                                              max_depth = param[1],       
                                              n_estimators = param[2],     
                                              criterion = param[3],        
                                              min_samples_leaf = param[4],
                                              min_samples_split = param[5],
                                              bootstrap = param[6])

        # Train model on 60% of the original data
        rf.fit(X_train2, y_train2)

        if (getOnlyModel == True):
            return (rf)
            break


        # Calculate predictions within training, dev, and test sets
        train_predictions   = rf.predict(X_train2)
        dev_predictions     = rf.predict(X_dev)
        test_predictions    = rf.predict(X_test)

        # Calculate accuracy on training,testing and dev sets
        from sklearn.metrics import accuracy_score
        trainAccuracy.append(accuracy_score(y_train2, train_predictions ))
        devAccuracy.append(accuracy_score(y_dev,dev_predictions))
        testAccuracy.append(accuracy_score(y_test,test_predictions))

        # Let us study the hyperparameters on the dev set
        actual      =   np.array(y_dev).flatten()
        predicted   =   dev_predictions.flatten()

        cm = confusion_matrix(actual, predicted)

        # Toggle in the funtion pass
        if (plotCM == True):
            plotCMatrix(cm)

        tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()

        # Calculate Classical Metrics of Model Performance and print
        p = tp / (fp + tp)
        r = tp / (tp + fn)
        f = 2 *(p * r) / (p + r)

        precision.append(p)
        recall.append(r)
        f1.append(f)
        times.append(time.time()-start)

    hyperDF = pd.DataFrame(hyperParameters)
    hyperDF.columns = ["max_features","max_depth","n_estimators",
                       "criterion","min_samples_leaf","min_samples_split","bootstrap"]
    accuracyDF = pd.DataFrame(data={'paramNumber': range(len(hyperParameters)), 'Train Accuracy': trainAccuracy, 'Test Accuracy': testAccuracy, 'Dev Accuracy': devAccuracy, 'Time':times})
    evaluateDF = pd.DataFrame(data={'paramNumber': range(len(hyperParameters)), 'Precision': precision, 'recall': recall, 'F1':f1, 'Time':times})

    finalDf = pd.concat([hyperDF.n_estimators,evaluateDF.iloc[:,1:-1],accuracyDF.iloc[:,1:]],axis=1)
    return(finalDf)


def plotXY(x,y):
    """
        :plotXY: Plot graph of x vs y

        :param x: List of values for X-axis
        :type x: List 

        :param y: List of values for y-axis
        :type y: List

        :returns: Plot graph of x vs y
    """
    xnew = np.linspace(y.min(), y.max(), 300) 

    spl = make_interp_spline(y, x, k=3)  
    power_smooth = spl(xnew)

    figure(num=None, figsize=(7,7), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(xnew, power_smooth)
    plt.title("F1 vs n_estimators",fontsize =15)
    plt.ylabel("F1",fontsize =15)
    plt.xlabel("n_estimators",fontsize =15)
    plt.legend()
    plt.show()
    
    
def plotCMatrix(cm):
    """
        :plotCMatrix: Plot confusion matrix

        :param cm: confusion matrix
        :type cm: np.array 

        :returns: Confusion matrix plot including TN,FP,FN,TP and percentages
    """
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    figure(num=None, figsize=(7,7), dpi=150)
    ax= plt.subplot()


    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm, ax=ax,annot=labels, fmt='', cmap='Blues')
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
