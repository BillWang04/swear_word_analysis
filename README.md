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

# Fair Warning, the following data project does contain the use of swear words and other offensive slurs as a part of the data.  

## Abstract

Since the start of the new millennium, the use of social media has continued to grow at an increasing rate as people have found themselves with the capabilities to express themselves online in a way that allows expressing their feelings to others without having to feel confined to a social setting where it can feel restrictive to what they can say. As this increasing use of social media has followed for many years, it has also come with many people being able to express things with curse words and slurs that vary in degrees of severity and range in different categories as to what they mean by these words. With the use of data sets found on popular data websites, we were able to find that there many times, the inflammatory words used online were meant to be in a way that described anatomy. We also gathered that the more common words by themselves were often slurs or common curse words said even in a social setting. Inside this data, we also were able to determine that more often than not, these inflammatory words reach a rating that would be deemed as 'strong', which was our highest severity ratings category used to judge these words. This project argues that the increasing use of social media has allowed people not only to express themselves online but also be much more severe in their word usage without the constant pressure of social setting standards, often use these words to express a manner that is seen as more sexual demeanor as their common reason.    

**TALK ABOUT THE CLASSIFICATION OF SWEAR WORDS TO CATEGORIES AND HOW THATS IMPORTANT (This is the main part of analysis in code)**

## Introduction

## About the Data

### Data Source

### About the Data

### EDA


<img src = "imgs/eda/most_common_swear_words.png">

<p align="center">
**Figure 1:Most Common Swear Words**
	</p>
</p>


<img src = "imgs/eda/avg_severity_rating.png">

<p align="center">
**Figure 2: Histogram Category Frequency**
	</p>
</p>


<p align="center">
<img src = "imgs/eda/frequency_of_each_category.png">

<p align="center">
**Figure 3: Histogram Category Frequency**
	</p>
</p>


<p align="center">
<img src = "imgs/eda/box_plot1.png">

<p align="center">
**Figure 4: Box Plot 1**
	</p>
</p>

<p align="center">
<img src = "imgs/eda/box_plot2.png">

<p align="center">
**Figure 5: Box Plot 2**
	</p>
</p>

<p align="center">
<img src = "imgs/eda/box_plot3.png">

<p align="center">
**Figure 6: Box Plot 3**
	</p>
</p>

<p align="center">
<img src = "imgs/eda/frequency_length_swear_words.png">

<p align="center">
**Figure 7: Frequency of the Length of Swear Words**
	</p>
</p>


<p align="center">
<img src = "imgs/eda/fws.png">

<p align="center">
**Figure 7: Frequency of Swear Words per Severity**
	</p>
</p>



# Predictive Model

## Preprocessing

1. **Data Preparation**: The dataset consists of texts of swear words, severity ratings, and a binary indicator for profanity hardness. 

2. **Train-Test Split**: The dataset is split into training and testing sets to train the model on one set and evaluate its performance on another.

3. **Feature Engineering**:
    - **N-gram Features**: N-grams are sequences of n items from the text. Bi-grams and tri-grams are extracted from the text samples, counting their occurrences to construct feature vectors.
    - **Severity Rating and Hardness Indicator**: Additional features include severity ratings and a binary indicator for profanity that end with a hard consenants.

4. **Label Binarization**: The target variable (profanity categories) is binarized to convert categorical labels into a binary matrix representation for later representation in Receiver Operating Characteristic (ROC) curve and Precision vs Recall Curve.

5. **Concatenation**: Extracted n-gram features, severity ratings, and hardness indicators are concatenated to form the final feature matrices for training and testing.



## Naive Bayes Model Report

Naive Bayes classifiers are based on Bayes' theorem with the assumption of conditional independence between every pair of features given the value of the class variable. This assumption simplifies the computation and is given by:

$$P(C \mid X) = \frac{P(C) \cdot P(X \mid C)}{P(X)}$$


In practice, we aim to maximize the posterior probability $P(C \mid X)$, and we often drop $P(X)$ as it is constant for all classes.

### Model Diagnostics

The Naive Bayes classifier achieved an accuracy of around 74%. Below are the detailed diagnostics including the confusion matrix, precision-recall curves, and One-vs-Rest multiclass ROC.

### Confusion Matrix
The confusion matrix shows the performance of the classification model on a set of test data for which the true values are known. The matrix illustrates the number of correct and incorrect predictions made by the model compared to the actual classifications in the test data.

![Confusion Matrix](imgs/classifier/naive_bayes_cm.png)

Because this a multiclass classifier, it is hard to purely disect the recall and percision of the classifier, and thus we must plot it.

### Precision-Recall Curve
The precision-recall curve is a plot of the precision (y-axis) versus the recall (x-axis) for a classifier at different threshold settings. It is useful for evaluating models when the classes are imbalanced.

![Precision-Recall Curve](imgs/classifier/naive_bayes_prc.png)
**Analysis:**

As one can see, there is wide variety of percision/recall strength when looking at the curve. Categories such as `religious offence`, `sexual_anatomy / sexual acts`, `racial / ethnic slurs`, and `sexual orientation / gender` perfrom reasonably well by maintaing a wide range of recall values, indicating that the model is effective in detecting these categories. However, categories of `bodily fluids / excrement`, `mental disability`, and `physical attributes` had moderate performance and `animal references`, `other / general insult`, `physical disability`, and `political` were poor or even missing from the graph. The missigness is a sign that there were no values that were predicted from that class, so there was no way to plot the Precision vs. Recall Curve

### One-vs-Rest multiclass ROC
The ROC curve, or Receiver Operating Characteristic curve, is a graphical representation of the true positive rate (TPR) versus the false positive rate (FPR) at various threshold settings. The area under the curve (AUC) is used as a measure of the model’s ability to distinguish between classes. An AUC of 1 indicates a perfect model, while an AUC of 0.5 suggests no discriminative ability. In this case, because we are multi-classed, we will use the One-vs-Rest multiclass ROC. In each step, a given class is regarded as the positive class and the remaining classes are regarded as the negative class as a bulk.

![ROC Curve](imgs/classifier/naive_bayes_roc.png)

**Analysis:**
The model demonstrated strong performance in distinguishing `racial/ethnic`, `religious offence` and `sexual anatomy / sexual acts`, indicating high discriminative power for these classes. On the other hand, the model showed moderate performance for classes such as`sexual orientation/gender`, `mental disability` and `bodily fluids`, suggesting room for improvement. The classifier struggled significantly with `physical attributes` and `general insults`, as its performance in these areas was quite poor, indicating a need for further refinement.

Moreover, there are some NAN values. This indicates that for those categoires, the true positive rate (TPR) and false positive rate (FPR) cannot be calculated, meaning there was no predictions in those categories, resulting in no ROC. 

The ROC curve shows that the classifier performs well overall, as indicated by the micro-average AUC of 0.88. However, there are variations in performance across different classes, with some classes having high AUC values and others having issues, as indicated by the nan AUC values. This suggests that the classifier may need improvement and more data for specific classes to achieve better performance uniformly.

Overall, the Naive Bayes classifier performed reasonably well, especially given the assumptions and the complexity of the task. It shows strong performance in certain categories such as `racial/ethnic `slurs and `sexual anatomy / sexual acts`, while struggling more with categories like `physical attributes` and `general insults`. The model's precision and recall values vary significantly across different classes, reflecting the challenges in correctly classifying some types of offensive content.



## Logestic Regression

After the perforance of Naive Bayes having the assumption of independence of each feature, we decided to use a different model, logestic regression. In the context of multiclass classification, logistic regression can be extended using the one-vs-rest (OvR) strategy. This approach involves breaking down a multiclass classification problem into multiple binary classification problems.

### Model Diagonistics

### Confusion Matrix and Precision-Recall Curve
![Confusion Matrix](imgs/classifier/lr_cm.png)


![Precision-Recall Curve](imgs/classifier/lr_prc.png)

Upon fitting the logestic regression model, the model came to be about a 83% accuracy. As you can see from the Precision vs. Recall curve,  there seems to show much better fit for classes that already performing fairly well in the Naive Bayes Model. For instance, `Religious Offenses` has a perfect 1 to 1 precision vs recall curve (). However, the increased metrics of the classes that already faired well does not seem to have increased the classes that were doing poorly. This is probably because there is not enough data for those classes.

### One-vs-Rest multiclass ROC

![ROC Curve](imgs/classifier/lr_roc.png)

It seems that the NaN values have not changed suggesting again that the predictions between these classes don't have enough data. However, looking the classes that are non NAN, other than `physical attributes`, the curves show a large AUC, suggesting strong discrimination between classes. 

### Conclusion

The Naive Bayes and Logistic Regression classifiers were compared to evaluate their performance on a multiclass classification task involving offensive content categorization. Each model's strengths and weaknesses were analyzed based on key metrics such as accuracy, confusion matrix, precision-recall curves, and One-vs-Rest ROC curves.

#### Naive Bayes Classifier
The Naive Bayes classifier, with its assumption of conditional independence, achieved an accuracy of around 74%. This model demonstrated strong performance in detecting certain categories like `religious offense`, `sexual anatomy / sexual acts`, and `racial / ethnic slurs`, as indicated by their robust precision-recall curves and high ROC AUC values. However, it struggled with categories such as `physical attributes`, `general insults`, and `political` references, often resulting in poor or even missing predictions for these classes. The variability in performance across different categories highlights the limitations of the Naive Bayes approach, particularly in handling the nuances of diverse and imbalanced data sets.

#### Logistic Regression Classifier
Similarly, the Logistic Regression classifier, using the One-vs-Rest (OvR) strategy, achieved a higher accuracy of 83%. This model showed improved precision-recall and ROC performance for the categories that already performed well under Naive Bayes, such as `religious offenses` and `sexual anatomy / sexual acts`. The AUC values for these categories were notably higher, indicating better discriminative power. However, similar to Naive Bayes, the Logistic Regression classifier also faced challenges with less represented categories like `physical attributes` and `general insults`, reflecting the need for more data to improve prediction accuracy for these classes.

#### Comparative Analysis
The Logistic Regression classifier generally outperformed the Naive Bayes model, particularly for well-represented classes. The ROC curves for Logistic Regression indicated stronger discriminative ability for most categories. However, both models exhibited limitations with poorly represented classes, which suggests that the primary issue is the imbalance and scarcity of data in these categories rather than the choice of the classifier.

### Recommendations
To enhance the performance of both classifiers, especially for underrepresented categories, the following steps are recommended:
1. **Data Augmentation:** Collect more data for the poorly performing categories to ensure a balanced representation.
2. **Feature Engineering:** Explore advanced feature engineering techniques to better capture the nuances of the text data.
3. **Hybrid Models:** Consider using ensemble methods that combine the strengths of different classifiers to improve overall performance.
4. **Regularization Techniques:** Apply regularization methods to handle data imbalance and prevent overfitting, particularly for Logistic Regression.

### Final Thoughts
The comparison underscores the importance of data quality and quantity in training robust classification models. While the Logistic Regression classifier showed superior performance overall, both models' effectiveness is contingent upon addressing data imbalances. Future efforts should focus on enriching the dataset and exploring hybrid modeling approaches to achieve more consistent and reliable classification across all categories.












