"""
@author: Shravankumar Hiregoudar
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def addDuplicateDF(df):
    """
        :addDuplicateDF: Function to add the duplicate features. This improves the results by increasing the records in the data
                         The A elements properties is swapped with B element properties to create more data. The duplicate
                         data is further concatenated with the original provided to us
        :param df: Original dataframe of features (Train / test dataframe)
        :type df: DataFrame

        :returns: Dataframe verison of features by including original and duplicate
    """
    # As contains all the element A properties 
    As = df.iloc[:, 0::2]

    # Checks for stability vector
    if 'stabilityVec' in df.columns:
        As = As.iloc[:,:-1]   
    Bs = df.iloc[:, 1::2] 

    # swap the column names of As and Bs
    As.columns, Bs.columns = Bs.columns, As.columns

    # Create the duplicate dataframe
    duplicate = pd.concat([As, Bs], axis=1, sort=False)

    # Return the final DF = original + duplicate
    if 'stabilityVec' in df.columns:
        t = (pd.concat([df.iloc[:,:-1],duplicate],sort = False,ignore_index=True))
    else:
        t = (pd.concat([df.iloc[:,:],duplicate],sort = False,ignore_index=True))

    return (t)

def addDuplicateTarget(target):
    """
        :addDuplicateTarget: Function to add the duplicate targets. The stability vector is reversed to make the duplicate verison
                             of elements. 
                             For example: If A/B stability vector (ignoring the pure element stability) is [0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0]
                             The function gives B/A stability vector i.e, [0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0]
        :param target: Stability vector
        :type target: list

        :returns: Final dataframe verison of targets by including original and duplicate
    """
    duplicateTarget = [elem[::-1] for elem in target]

    result = sum([target, duplicateTarget], [])
    return(pd.DataFrame(result).astype(int))

def addFeatures(df):
    """
        :addFeatures: Adding features from the existing features of train / test data. For example, avg atomic number, diff in atomic number, etc.
        :param df: Dataframe of features after adding duplicates (Train / test dataframe)
        :type df: DataFrame

        :returns: Dataframe verison2 of features by including additional features
    """
    As = df.iloc[:,2:-4].iloc[:, 0::2]
    Bs = df.iloc[:,2:-4].iloc[:, 1::2] 

    for i in range(As.shape[1]):
        df['avg_'+As.columns[i].split("A_")[1]] = df.loc[: , As.columns[i]:Bs.columns[i]].mean(axis=1)
        df['diff_'+As.columns[i].split("A_")[1]] = As.iloc[:,i] - Bs.iloc[:,i]
        
    return (df)

def rfFeatureSelection(features, target, summary = False, plot = True):
    """
        :rfFeatureSelection: Random Forest feature selection, Plot the feature number vs Feature Importances by fitting RF. 
                             This helps us to understand the importance of each of properties of elements.
                             NOTE: The gini index is considered with default number of trees as 100
        :param features: Training features data
        :type features: DataFrame

        :param target: Training target data
        :type target: DataFrame

        :param summary: If true, Return the dataframe containing feature number and corresponding importance 
        :type summary: bool

        :param plot: If true, Display graph of feature number vs feature importance
        :type plot: bool

        :returns: Graph of feature number vs Feature Importance for target based on RF
    """
    # define the model
    model = RandomForestClassifier()
    # fit the model
    model.fit(features, target)
    # get importance
    importance = model.feature_importances_
    
    # Plot toggle
    if plot == True:
        # plot feature importance
        fig, ax = plt.subplots(figsize=(20,10), dpi=150)
        plt.bar([x for x in range(len(importance))], importance)
        plt.axhline(y=0.005,color = 'black')
        ax.set_xticks(np.arange(0,(len(importance)),5))
        ax.set_ylabel("RF Imoprtance Score",fontsize=20)
        ax.set_xlabel('Features',fontsize=20)
        ax.set_title('Random Forest Feature Selection',fontsize=20)
        plt.show()
        
    # Summary toggle
    if summary == True:
        df = pd.DataFrame({'features' : [x for x in range(len(importance))], 'importance' : importance})
        return df

def getGoodIndex(summary,thresholdImportance=0.005):
    """
        :getGoodIndex: Get the index numbers of all the features for which RF gave importance greater than threshold.
                       Only in case of property of both A and B are greater than threshold, Consider both indices as good indices
                       Note: If importance of property of A > 0.005 but B < 0.005, Then both indices are not considered.
        :param summary: Dataframe containing feature number and corresponding importance
        :type summary: DataFrame

        :param thresholdImportance: Threshold value of feature importance for considering the feature as good feature
        :type thresholdImportance: float

        :returns: List of good indices based on the RF feature importance 
    """
    # Get indices having importance greater than threshold
    importantFeatures = np.where(summary.importance > thresholdImportance)[0]

    # Indices of all the properties of A
    AProperties = list(filter(lambda x: (x % 2 == 0), importantFeatures)) 
    goodIndex = []

    # Check if both properties of A and B is greater than threshold
    for i in range(len(AProperties)):
        if AProperties[i]+1 in importantFeatures:
            goodIndex.extend((AProperties[i],AProperties[i]+1))

    return (goodIndex)