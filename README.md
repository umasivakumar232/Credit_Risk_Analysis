# Credit_Risk_Analysis

## Overview

In this assignment we will be helping Jill apply machine learning to solve a real-world challenge: **credit risk**. 

## Machine Learning 

Machine learning comprises a broad range of analytical tools, which can be categorized into **“supervised”** and **“unsupervised”** learning tools. Supervised machine learning involves building a statistical model for predicting or estimating an output based on one or more inputs.  In unsupervised learning, a dataset is analyzed without a dependent variable to estimate or predict.  

## Machine learning methods

The machine learning spectrum comprises many different analytical methods, whose applicability varies with the types of statistical  problem  one  might want to  address. Broadly speaking, machine learning can be applied to three classes of statistical  problems:  **regression**,  **classification**,  and  **clustering**. Regression and classification problems both can be solved through supervised machine learning. We will be using these methods in this assignment. 


## Prediction versus explanation 

Machine learning’s ability to make out-of-sample predictions does not necessarily make it appropriate for explanation or inferences. A good predictive model can be very complex, and may thus be  very  hard  to  interpret. For  predictive  purposes,  a  model would need only to give insight in correlations between variables, not in causality. In the case of credit scoring a loan portfolio, a good inferential model would explain why certain borrowers do not repay their loans. Its inferential performance can be assessed through its statistical significance and its goodness-of-fit within the data sample. A good predictive model, on the other hand, will select those indicators that prove to be the strongest predictors of a borrower default. To that end, does not matter whether an indicator reflects a causal factor of the borrower’s ability to repay, or a symptom of it. What matters is that it contains information about the ability to repay.

## Typical pitfalls and ways to deal with them

Excessively complex models can also lead to **“overfitting,”** where they describe random error or noise instead of underlying  relationships  in  the  dataset.  When a model describes noise in a dataset, it will fit that one data sample very well, but will perform poorly when tested out-of-sample.There are several ways to deal with overfitting and improve the forecast power of machine learning models, including **“bootstrapping,”** **“boosting”** and **“bootstrap aggregation”** (also called bagging). 

**Boosting** concerns the overweighting of scarcer observations in a training dataset to ensure the model will train more intensively on them. For example, one may want to overweight the fraudulent observations due to their relative scarcity  when  training  a  model  to  detect  fraudulent  transactions in a dataset.  

**“bagging,”** a model is run hundreds or thousands of times, each on a different subsample of the dataset,  to  improve  its  predictive  performance.  The  final  model  is then an average of each of the run models

## Resources

Software: Jupyter Notebooks

Languages: Python

Libraries: numpy, pandas, matplotlib, scikit-learn, imbalance-learn

Data Sources: Loan Data: LoanStats_2019Q1.csv

## Analysis

Credit risk is an inherently unbalanced classification problem, as **good loans** easily outnumber **risky loans**. Therefore, we need to employ different techniques to train and evaluate models with unbalanced classes. A Look at our data revels that we have information of 68470 low risk instances and only 347 high risk instances

We will use and compare  various techniques such as resampling  and ensemble classifiers to see which one technique will help us predict the High risk instances better.  

Resampling techniques include - 

* Oversampling the data using the **RandomOverSampler** and **SMOTE algorithms**
* Undersampling the data using the **ClusterCentroids** algorithm. 
* Combinatorial approach of over-and undersampling using the **SMOTEENN algorithm**. 

Ensemble Classifiers 

* BalancedRandomForestClassifier 
* EasyEnsembleClassifier

### Using Resampling Techniques to Predict Credit Risk 

Evaluate three machine learning models by using resampling to determine which is better at predicting credit risk

#### Oversampling - RandomOverSampler

In random oversampling, instances of the minority class **(high risk loans)** are randomly selected and added to the training set until the majority and minority classes are balanced 

<img width="394" alt="RandomOverSampler" src="https://user-images.githubusercontent.com/85518330/136819953-2e30a02d-2b50-4da6-9cc1-75e08a06a98b.png">


<img width="501" alt="RandomOversampling1" src="https://user-images.githubusercontent.com/85518330/136819603-ade2fc11-7184-4790-a488-19a67748231f.png">

##### Interpretations of the results - RandomOverSampler

A look at the imbalance classification report above indicates that this model is only able to correctly predict 2 out of 101 high risk instances. This is indicated by the F1 score of 0.02.  F1 score is the weighted harmonic mean of precision and recall where the best score in 1 and worst is 0.0. The RandomOversampler method is quite inadequate in this case 

#### Oversampling - SMOTE Oversampling 

The synthetic minority oversampling technique (SMOTE) is another oversampling approach. Here new instances are interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.

<img width="581" alt="SMOTE" src="https://user-images.githubusercontent.com/85518330/136822363-b48413dd-1aed-463d-9a31-d51095611490.png">

##### Interpretations of the results - SMOTE Oversampling

A look at the imbalance classification report indicates that this model also had a F1 score of 0.02 for high risk instances, however the F1 score of low risk instances improved from 0.75 in RandomOverSampler to 0.82 here. Our purpose however is to be able to predict high risk cases better so this is also inadequate for our purpose

#### Undersampling - ClusterCentroids

Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased.
Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class

<img width="565" alt="undersampling" src="https://user-images.githubusercontent.com/85518330/136824657-1aec2b33-5930-462c-aeda-9177125e688e.png">

<img width="513" alt="ClusterCentroids" src="https://user-images.githubusercontent.com/85518330/136824685-af2a1472-eced-4cf6-b9b3-c99ef8d0e202.png">

##### Interpretations of the results - ClusterCentroids Undersampling

A look at the imbalance classification report indicates that the cluster centroid method results are poorer than the oversampler method results in predicting both high risk as well as low risk instances. Indicated by an F1 score of 0.01 for high risk instances and 0.57 for low risk instances 


#### Combination Sampling  - SMOTEENN

SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process
* Oversample the minority class with SMOTE.
* Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.

<img width="575" alt="Smoteenn" src="https://user-images.githubusercontent.com/85518330/136825978-02d258eb-162d-4e03-9adb-39c302a2c872.png">

<img width="495" alt="Smoteenn1" src="https://user-images.githubusercontent.com/85518330/136830950-4fc87e9a-7278-4de4-a88d-6cba4ee389ba.png">

##### Interpretations of the results - SMOTEENN

A look at the imbalance classification report indicates that the SMOTEENN method results are similar to that of the Oversampling methods in predicting both high risk as well as low risk instances. Indicated by an F1 score of 0.02 for high risk instances and 0.72 for low risk instances 

### Overall Results from resampling methods

Overall the results of all the resampling methods were poor for predicting high risk instances.

### Using Ensemble Classifiers to Predict Credit Risk

The concept of ensemble learning is the process of combining multiple models, like decision tree algorithms, to help improve the accuracy and robustness, as well as decrease variance of the model, and therefore increase the overall performance of the mode. Here we will compare two ensemble algorithms to determine which algorithm results in the best performance. we train a **Balanced Random Forest Classifier** and an **Easy Ensemble AdaBoost classifier**. 

#### Ensemble classifier - Balanced Random Forest Classifier

A balanced random forest randomly under-samples each boostrap sample to balance it. 

<img width="505" alt="BalancedRandomForest" src="https://user-images.githubusercontent.com/85518330/136834463-adf34e6d-ffd9-482e-bddf-cd7cf8471e62.png">

<img width="509" alt="BalancedRandomForest1" src="https://user-images.githubusercontent.com/85518330/136834488-039339d1-17a8-4651-86d9-4eadf672e57e.png">


##### Interpretations of the results - Balanced Random Forest Classifier

A look at the imbalance classification report indicates that the BRF classifier method results were so much better than the ones we got from the resampling methods. Here the F1 score of high risk instances went up to 0.06 and the low risk instances went up to 0.93.


#### Ensemble classifier - Easy Ensemble ADABoost Classifier

Adaboost uses stumps (decision tree with only one split). Adaboost is basically a forest of stumps. These stumps are called weak learners. These weak learners have high bias and low variance. Stumps use one feature at a time

<img width="529" alt="ADAboost" src="https://user-images.githubusercontent.com/85518330/136838220-d386fe66-4617-442c-97fd-f9674ef31ba5.png">

##### Interpretations of the results - Easy Ensemble ADABoost Classifier

A look at the imbalance classification report indicates a vast improvement in the results when we use the easy ensemble ADABoost classifier. Here the F1 score of high risk instances went up to 0.16 and the low risk instances went up to 0.97.

## Conclusions and Recommendations

From the preformance of all the models it is quite clear that the **Easy Ensemble ADABoost Classifier** results were better than all the other techniques for prediciting High Risk loans. Aspects of this technique that worked in our favor are the following 

- Combining weak learners into a strong learner. This helps reduce the overfitting issue that such models generally may have.
- The weak learners are not combined at the same time. Instead, they are used sequentially, as one model learns from the mistakes of the previous model.

