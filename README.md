# Predict-stability-of-chemical-comounds

The idea behind this challenge is to use data to solve a canonical thermodynamics problem: given a pair of elements, predict the stable binary compounds that form on mixing. Within the attached .zip, we've provided roughly 5000 element pairs as training data. Each of the pure elemental compounds have been expanded into features for you using a naive application of the magpie feature set (https://bitbucket.org/wolverton/magpie). Feel free to prune, extend, or otherwise manipulate this feature set in pursuit of a predictive model!

The training labels are a discretization of the 1D binary phase diagram at 10% intervals. For example, the label for OsTi ([1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0]) translates into the following stable compounds:  Os, Os{0.5}Ti{0.5} or OsTi, and Ti. 

Task is to build a machine learning model in Python to predict the full stability vector.  


**Read report in results directory for complete picture of the project**
