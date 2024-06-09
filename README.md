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

Since the start of the new millennium, the use of social media has continued to grow at an increasing rate as people have found themselves with the capabilities to express themselves online in a way that allows expressing their feelings to others without having to feel confined to a social setting where it can feel restrictive to what they can say. As this increasing use of social media has followed for many years, it has also come with many people being able to express things with curse words and slurs that vary in degrees of severity and range in different categories as to what they mean by these words. The main interest of our data analysis was to see if we could analyze what was the general purpose that people would intend their use of inflammatory language online, being able to see from our data source "profanity.csv" how each of these swear words was categorized, with this being the best way to gauge the general direction of the usage. We also gathered that the more common words by themselves were often slurs or common curse words said even in a social setting. Inside this data, we also were able to determine that more often than not, these inflammatory words reach a rating that would be deemed as 'strong', which was our highest severity ratings category used to judge these words. This project argues that the increasing use of social media has allowed people not only to express themselves online but also be much more severe in their word usage without the constant pressure of social setting standards, often use these words to express a manner that is seen as more sexual demeanor as their common reason.    


## Introduction

In recent years, the rise of social media platforms has revolutionized the way people communicate, providing an unprecedented level of freedom in expressing thoughts and emotions. However, this freedom has also led to the increased use of offensive language online, including curse words and slurs, which can vary significantly in severity and context. The pervasive use of such language poses challenges for maintaining respectful and safe online communities. To address these challenges, this research project aims to analyze the structural features of swear words and develop a model for efficient content filtering.

## About the Data

### Data Source
The data for this research project is sourced from the ["Collection of Profanities in English"](https://www.kaggle.com/datasets/konradb/profanities-in-english-collection) available on Kaggle. This dataset includes a wide range of English profanity words categorized by severity and type. The dataset aims to support the development of content moderation tools by providing a comprehensive list of offensive language.

### About the Data
This dataset contains over 1600 popular English profanities and their variations. The columns in the dataset include:

- text: The profanity word or phrase
- canonical_form_1: The primary canonical form of the profanity.
- canonical_form_2: An additional canonical form, if applicable.
- canonical_form_3: Another additional canonical form, if applicable.
- category_1: The profanity's primary category.
- category_2: The profanity's secondary category, if applicable.
- category_3: The profanity's tertiary category, if applicable.
- severity_rating: The average severity rating of the profanity, based on a scale from 1 to 3.
- severity_description: The description of the severity rating, rounded to the nearest integer and categorized as Mild (1), Strong (2), or Severe (3).

The profanities are organized into several categories to help understand their context and usage. These categories include:
- Sexual Anatomy / Sexual Acts
- Bodily Fluids / Excrement
- Sexual Orientation / Gender
- Racial / Ethnic Slurs
- Mental Disability
- Physical Disability
- Physical Attributes
- Religious Offense
- Political
- Other / General Insult

### EDA


<img src = "imgs/eda/most_common_swear_words.png">

<p align="center">
**Figure 1:Most Common Swear Words**
	</p> Data Analysis: This data from this graph indicates how "Fuck" is by far the most commonly used inflammatory word documented as the following words like "motherfucker" and "shit" follow next, with this graph looking similar to an exponential decay graph.
</p>


<img src = "imgs/eda/avg_severity_rating.png">

<p align="center">
**Figure 2: Histogram Category Frequency**
	</p> Data Analysis: The chart shows that insults targeting race, physical disability, and political affiliations are perceived as the most severe, being the ones well above the mean of the overall data, hanging at around a 2.5 average severity rating. In contrast, insults involving animal references and physical attributes are perceived as the least severe as they almost average a severity rating of 1.
</p>


<p align="center">
<img src = "imgs/eda/frequency_of_each_category.png">

<p align="center">
**Figure 3: Histogram Category Frequency**
	</p> Data Analysis: 
</p>


<p align="center">
<img src = "imgs/eda/box_plot1.png">

<p align="center">
**Figure 4: Box Plot 1**
	</p> Data Analysis:
</p>

<p align="center">
<img src = "imgs/eda/box_plot2.png">

<p align="center">
**Figure 5: Box Plot 2**
	</p> Data Analysis:
</p>

<p align="center">
<img src = "imgs/eda/box_plot3.png">

<p align="center">
**Figure 6: Box Plot 3**
	</p> Data Analysis:
</p>

<p align="center">
<img src = "imgs/eda/frequency_length_swear_words.png">

<p align="center">
**Figure 7: Frequency of the Length of Swear Words**
	</p> Data Analysis:
</p>


<p align="center">
<img src = "imgs/eda/fws.png">

<p align="center">
**Figure 8: Frequency of Swear Words per Severity**
	</p> Data Analysis:
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

All in all, it seems that the one-vs-rest logestic regression classifier exceeded the Naive Bayes with classes with sufficient data. If you recall from *Figure 3*, the swear words are a majority in the class `sexual_anatomy / sexual acts` (980/ 1598 = 61%) while the top 5 categories consist of 1552/1598 = 97%. Thus, if we want a stronger prediction for the rest of the classes, we should get more data on those classes. 













