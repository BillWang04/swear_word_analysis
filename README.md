# Swear Word Analysis


<p align="center">
Bill Wang,
</p>

<p align="center">
Andrew Vela,
</p>

<p align="center">
Songjiang Zhou,
</p>

## Abstract

## Introduction

## About the Data

### Data Source

### About the Data

### EDA

## Predictive Model

### Preprocessing

1. **Data Preparation**: The dataset consists of texts of swear words, severity ratings, and a binary indicator for profanity hardness. 

2. **Train-Test Split**: The dataset is split into training and testing sets to train the model on one set and evaluate its performance on another.

3. **Feature Engineering**:
    - **N-gram Features**: N-grams are sequences of n items from the text. Bi-grams and tri-grams are extracted from the text samples, counting their occurrences to construct feature vectors.
    - **Severity Rating and Hardness Indicator**: Additional features include severity ratings and a binary indicator for profanity hardness.

4. **Label Binarization**: The target variable (profanity categories) is binarized to convert categorical labels into a binary matrix representation for later representation in Receiver Operating Characteristic (ROC) curve and Precision vs Recall Curve.

5. **Concatenation**: Extracted n-gram features, severity ratings, and hardness indicators are concatenated to form the final feature matrices for training and testing.


### Naive Bayes Model Report

#### Assumptions
Naive Bayes classifiers are based on Bayes' theorem with the assumption of conditional independence between every pair of features given the value of the class variable. This assumption simplifies the computation and is given by:

$$ P(C \mid X) = \frac{P(C) \cdot P(X \mid C)}{P(X)} $$


In practice, we aim to maximize the posterior probability $ P(C \mid X) $, and we often drop $ P(X) $ as it is constant for all classes.

#### Model Diagnostics

The Naive Bayes classifier achieved an accuracy of around 74%. Below are the detailed diagnostics including the confusion matrix, precision-recall curves, and One-vs-Rest multiclass ROC.

#### Confusion Matrix
The confusion matrix shows the performance of the classification model on a set of test data for which the true values are known. The matrix illustrates the number of correct and incorrect predictions made by the model compared to the actual classifications in the test data.

![Confusion Matrix](imgs/classifier/naive_bayes_cm.png)

As you can see, because most of the data is categorized as `sexual anatomy/sexual acts`, we can see that most predictions are of the 

#### Precision-Recall Curve
The precision-recall curve is a plot of the precision (y-axis) versus the recall (x-axis) for a classifier at different threshold settings. It is useful for evaluating models when the classes are imbalanced.

![Precision-Recall Curve](imgs/classifier/naive_bayes_prc.png)
**Analysis:**

As one can see, there is wide variety of percision/recall strength when looking at the curve. Categories such as `religious offence`, `sexual_anatomy / sexual acts`, `racial / ethnic slurs`, and `sexual orientation / gender` perfrom reasonably well by maintaing a wide range of recall values, indicating that the model is effective in detecting these categories. However, categories of `bodily fluids / excrement`, `mental disability`, and `physical attributes` had moderate performance and `animal references`, `other / general insult`, `physical disability`, and `political` were poor or even missing from the graph. The missigness is a sign that there were no values that were predicted from that class, so there was no way to plot the Precision vs. Recall Curve

#### One-vs-Rest multiclass ROC
The ROC curve, or Receiver Operating Characteristic curve, is a graphical representation of the true positive rate (TPR) versus the false positive rate (FPR) at various threshold settings. The area under the curve (AUC) is used as a measure of the model’s ability to distinguish between classes. An AUC of 1 indicates a perfect model, while an AUC of 0.5 suggests no discriminative ability. In this case, because we are multi-classed, we will use the One-vs-Rest multiclass ROC. In each step, a given class is regarded as the positive class and the remaining classes are regarded as the negative class as a bulk.

![ROC Curve](imgs/classifier/naive_bayes_roc.png)

**Analysis:**
The model demonstrated strong performance in distinguishing `racial/ethnic`, `religious offence` and `sexual anatomy / sexual acts`, indicating high discriminative power for these classes. On the other hand, the model showed moderate performance for classes such as`sexual orientation/gender`, `mental disability` and `bodily fluids`, suggesting room for improvement. The classifier struggled significantly with `physical attributes` and `general insults`, as its performance in these areas was quite poor, indicating a need for further refinement.

Moreover, there are some NAN values. This indicates that for those categoires, the true positive rate (TPR) and false positive rate (FPR) cannot be calculated, meaning there was no predictions in those categories, resulting in no ROC. 

The ROC curve shows that the classifier performs well overall, as indicated by the micro-average AUC of 0.88. However, there are variations in performance across different classes, with some classes having high AUC values and others having issues, as indicated by the nan AUC values. This suggests that the classifier may need improvement and more data for specific classes to achieve better performance uniformly.

Overall, the Naive Bayes classifier performed reasonably well, especially given the assumptions and the complexity of the task. It shows strong performance in certain categories such as `racial/ethnic `slurs and `sexual anatomy / sexual acts`, while struggling more with categories like `physical attributes` and `general insults`. The model's precision and recall values vary significantly across different classes, reflecting the challenges in correctly classifying some types of offensive content.



### Logestic Regression

#### Assumptions

#### Model Diagonistics
