# Kaggle AdTracking Competition

This is my entry to the competition. The problem tackled is a very modern one and crucial to our current economy : how to prevent click-fraud in the onlive advertisment industry. The chinese firm AdTracking offers advertisment to clients through many channels. It pays these channels on click and in turn is paid by the client for the exposure they received, counted by engagement, in other words clicks. Some bad actors in this system generate clicks that do not correspond to actual potential customers and get revenue for it. It is often called "click farm".

  The challenge here is not actually to identify these bad actors but to predict clicks that result in actual engagement. The target feature, called is_attributed, is a boolean indicating if the click was followed by a download of the advertised app. The problem is therefore a binary classification.

  The evaluation metric chosen for the competition the **roc_auc**, generally known as "Area Under the Curve".

  Reference: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

  **// Dataset description and optimization//**

  This dataset is far from being BIG DATA, but being a few hundred million rows, the loaded raw data fills almost 4Gb of memory. Yet it is still one of the biggest datasets in Kaggle competitions. What that meant for me is that very tight optimization needed to be done to function locally (8Gb of RAM), and that Kaggle kernels would only go so far with its 17Gb.

  You can find some EDA (Exploratory Data Analysis) here:
    - link 1
    - link 2
    - link 3

  The original features (click_time, ip, app, channel, os, device) were all, beside the time stamp and target, categorical. 


**// Model Selection //**

To my knowledgem the models available to us for binary classification are:
* Logistic Regression
* Decision Trees and their ensembles : Random Forest, Gradient Boosted Trees (XGBoost, Lightgbm, CatBoost)
* Support Vector Classifier
* Naive Bayse Classifiers (poor at probability prediction, so low performance on roc_auc)
* Neural Networks

Since I do not have the ressources for Neural Nets and that SVM and Logistic Regressions will give subpar performance due to the high non-continuity of the data, I picked decision tree based techniques for the competition. At first, I went for the underdog - Catboost.

Catboost (link) is the newest of the Gradient Boosting Machines listed above and comes from the russian firm Yandex. It is leading performance and speed in most benchmarks. It also specializes in handling categorical features. It sounded like the perfect candidate for the competition. Catboost came with its own challenges though as the documentation is much less extensive than the other two and more importantly it is **extremely** glutonous in memory. So much so that training a 4Gb dataset requires over 66Gb of memory. I favored Microsoft's lightGBM (link) instead.

**// Categorical Features //**

At first glance, it seems that our 5 features are categorical variables. They have been anonymized so the variable type is integers. No indication was given as to how the encoding was done and a quick EDA (link) told us some very surprizing information about the distribution of the values. In any case, a choice needs to be made for each of them. We can:
* Method 1 - keep them as integers, possibly re-ordering them by some metric (what catboost does).
* Method 2 - Use One-Hot-Encoding, creating as many boolean features as there are distinct values (minus one).
* Method 3 - Use Entity Embeddings.

Moreover, the target feature is extremely unbalanced. about 0.2% of all rows are pisitive. This adds in complexity and there are also different ways to deal with the distribution imbalance of the target feature:
* Fix 1 - Undersampling
* Fix 2 - Random Oversampling (boostrap of the positive class)
* Fix 3 - Clustering Oversampling
* Fix 4 - Synthetic Minority Oversampling Technique (SMOTE)
* Fix 5 - Modified Synthetic Minority Oversampling Technique (MSMOTE)

(The last 3 may not make sense with categorical features, see https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/ for details, moreover another contestant's analysis (link) shows that there is no gain to them.)

Ideally, we want to also compute the run time of each of these models to have a measure of the marginal cost in performance gains, so as to make buiseness sense of it once the competition is over.

**// Feature Engineering //**

Improving the predictive power of a given model requires to either feed more data or better data, which translates to either find relevant external data sources or craft some new features out of the existing ones. Considering the anonymity of the given dataset, there is no possibility to use external data sources, so we can focus our attention towards new features could be crafted and which model it would benefit:

* Time windows , e.g. morning/afternoon/evening.
* Group-split-combine aggregates.
* Moving sums and/or averages of the target variable for different features.
* Moving sums and probabilities of different features
* Non linear variables - for linear models (logistic regression and SVC) (need numeric features)
* other time series based variables.

**// A first step towards production quality code //**