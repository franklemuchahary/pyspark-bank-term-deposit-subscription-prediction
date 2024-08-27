# Team Mini Project: Bank Term Deposit Subscription Prediction

## Executive Summary
This project addressed the challenge of predicting potential subscribers for a bank's term deposit plan. The primary goal was to enhance the bank's marketing strategy through data-driven insights. We approached this by:

- **Problem Statement:** Identifying clients most likely to subscribe to term deposits, thereby optimizing marketing efforts and resource allocation.

- **Methods:** 
    - **Data Exploration with PySpark:** Conducting a thorough exploratory data analysis (EDA) to understand underlying patterns and correlations in the dataset.
    - **Modeling with PySpark:** Employing PySpark for efficient handling of large datasets and implementing various machine learning algorithms.
    - **Cluster Analysis with PySpark:** Segmenting clients into distinct groups based on their characteristics to better target marketing efforts.

- **Model Selection and Performance:**
    - The Gradient Boosting Model was selected for its superior performance.
    - **Key Performance Metrics:** The model achieved an 94% AUC-ROC, reflecting its effectiveness in classifying potential subscribers.

- **Key Findings:**
    - Duration of customer interaction and economic indicators (like Euribor rates) were significant factors.
    - Cluster analysis revealed distinct client segments with focus on old successful customers and old low effort customers that have high conversion rate, enabling tailored marketing strategies.

- **Recommendations:**
  - **Focus on Existing Customers:** Our findings show that existing customers have a higher average subscription rate. They require fewer contacts and less time on calls, suggesting that resources should be allocated towards maintaining and upselling to the current client base.

  - **Quality over Quantity:** New customers respond better to quality engagement. Therefore, we recommend investing in crafting effective sales pitches and utilizing engaging infographics. This approach should aim to deliver the most relevant and personalized information to new clients, enhancing the chance of subscription.

  - **Frequent Customer Contact:** Keeping in touch with interested customers is crucial, as they show a higher likelihood of subscribing. Regular and meaningful engagement can ensure that the customers' interest does not wane, increasing the chances of conversion.


These findings provide a roadmap for the bank to refine its approach to marketing term deposits, ensuring more targeted and effective client engagement.

## File Directory
- [EDA.ipynb](./Code/EDA.ipynb): This Jupyter notebook conducts an exploratory data analysis (EDA) on the XYZ_Bank_Deposit_Data_Classification.csv dataset using PySpark. Key tasks include:
    - Basic data understanding
    - Checking for missing values
    - Univariate analysis
    - Bivariate analysis
    - Correlation matrix
- [PredictiveModeling&SegmentationCode.ipynb](./Code/PredictiveModeling&SegmentationCode.ipynb): This notebook covers the preprocessing, train-test split, and modeling phases, using logistic regression, random forest, support vector classifier, gradient boosting classifier, and decision trees. It also includes customer segmentation analysis.
- [clustering_output_6clusters.csv](./Clustering%20Output/clustering_output_6clusters.csv): Stores the output of the clustering analysis with six clusters.
- [Predictive Models Folder](./Predictive%20Models): Stores all predictive models developed during the project.
- [Mini Project 2 (Team 3) - Final Presentation.pptx](./Mini%20Project%202%20(Team%203)%20-%20Final%20Presentation.pptx): Stores the final presentation outlining the project results and insights.


## Team Members
- Roy Chowdhury, Ritwik
- Datta Chakraborty, Shreyan
- Muchahary, Frankle
- Tan, Roger

## Introduction
The aim was to analyze the dataset's variables to predict clients' likelihood of subscribing to term deposits, utilizing PySpark for data processing.

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Modeling](#modeling)
5. [Recommendations](#recommendations)
6. [Conclusion](#conclusion)

## Exploratory Data Analysis (EDA)
- Analysis of 'Unknown' categories showing “Unknown” categories could indicate data discrepancies.
- Construction of a correlation matrix showing duration of calls, number of employees, and pdays (number of days that passed by after the client was last contacted from a 
previous campaign) are highly correlated with the target variable.
- Multiple univariate and bivariate analyses to explore distributions and relationships between variables.

## Modeling
- Data preprocessing including data transformation and standardization and application of machine learning models, including Logistic Regression, Decision Trees, and Gradient Boosting.
- Cross validation and model parameter tuning and optimization.

### Link to Model Files
- [Gradient Boosting Tree CV Model](https://github.com/shreyanDC/Mini-Project-2---BAN-5753-Team-3/tree/main/Predictive%20Models/gbt_cv_model)
- [Logistic Regression CV Model](https://github.com/shreyanDC/Mini-Project-2---BAN-5753-Team-3/tree/main/Predictive%20Models/logistic_regression_cv_model)
- [Random Forest CV Model](https://github.com/shreyanDC/Mini-Project-2---BAN-5753-Team-3/tree/main/Predictive%20Models/rf_cv_model)
- [Support Vector Classifier CV Model](https://github.com/shreyanDC/Mini-Project-2---BAN-5753-Team-3/tree/main/Predictive%20Models/svc_cv_model)
- [Vanilla Decision Tree CV Model](https://github.com/shreyanDC/Mini-Project-2---BAN-5753-Team-3/tree/main/Predictive%20Models/vanilla_dt_cv_model)

### Model Comparison
- Evaluation of models based on metrics like accuracy, precision, recall, F1-score, and ROC curves.
- Gradient boosting model appears as the best model with AUC-ROC of 94%

### Important Features 
- Key Features include a mix of internal factors and external macro-economic factor
- Duration of interaction turns out to be the key differentiator in customer conversion
- The Euribor Rate (3 Months) is also an important indicator of market conditions and ultimately help in determining customer conversion

## Clustering
The client base was segmented into five distinct groups based on multiple factors which included subscription rates, average call duration, and the number of times contacted. Here is a summary of the clusters:

- **New Unsuccessful Customers:** Characterized by a low average subscription rate of 7%. They had short call durations averaging 4 minutes and were contacted approximately 2.5 times.

- **Moderately Successful New Customers:** This group had a higher subscription rate of 39.7%, with an average call duration of around 6 minutes and contact frequency of about 2.4 times.

- **Relatively New & Interested Customers (High Effort):** Showed a subscription rate of 58.7% for new customers, with much longer average call durations of approximately 24 minutes and a contact frequency of roughly 2.6 times.

- **Existing Successful Customers:** The most successful group with a subscription rate of 80.5%, they required minimal effort with average call durations of 6 minutes and were contacted the least, around 1.8 times.

- **Existing Low Effort Customers:** Had a relatively high subscription rate of 60.2% despite low average call durations of less than 4 minutes and a contact frequency of about 1.82 times.

These clusters indicate varying levels of engagement and success, suggesting the need for tailored marketing strategies for different segments to optimize conversion rates.


## Recommendations
3-Pronged Recommendation Strategy:

- Focus on Existing Customers: Higher Average Subscription Rate achieved with lesser number of contacts and lower average call duration. 
- Quality over Quantity: Invest in effective sales pitches and infographics to convince new customers with most relevant and personalized information. 
- Frequent Customer Contact: Stay in touch with interested customers as they have a relatively higher likely likelihood of subscribing. Don’t let the trail go cold. 

## Conclusion
The project's analytical journey culminated in a robust Gradient Boosting Model that effectively predicted term deposit subscriptions with a 94% AUC-ROC score. This high level of accuracy underscores the model's capability in distinguishing between potential subscribers and non-subscribers.

Key takeaways based on our findings are:

- **Optimizing Engagement Strategies:**
  - **Old Successful Customers:** With an impressive 80% conversion rate, dedicated campaigns to retain and engage these customers should be a priority.
  - **Old Low Effort Customers:** Having a 60% conversion rate, this group could be targeted more efficiently, maximizing outcomes with minimal investment.

- **Leveraging Key Factors:**
  - **Internal Factor - Duration of Interaction:** Our analysis suggests that longer durations of customer interaction are pivotal in increasing the likelihood of subscription.
  - **External Factor - Euribor Rate (3 Months):** The model indicates that external economic conditions, as reflected by the Euribor rate, significantly influence subscription decisions.

In conclusion, the insights gained from our model provide a data-driven foundation for strategic decision-making in marketing term deposit subscriptions. By focusing on the identified key factors and target groups, banks can significantly enhance the efficacy and efficiency of their marketing strategies.

---

For more detailed information or collaboration, please contact any of the team members.
