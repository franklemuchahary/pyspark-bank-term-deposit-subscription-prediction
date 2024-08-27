#!/usr/bin/env python
# coding: utf-8

# # Codes for EDA
# 
# This notebook explores the dataset of `XYZ_Bank_Deposit_Data_Classification.csv` including 
# - Basic data understanding
# - Check for missing values
# - Univariate analysis
# - Bivariate analysis
# - Correlation matirx
# - Transformation Check

# In[1]:


import findspark
findspark.init()

# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import * 
import pyspark.sql.functions as F
from pyspark.sql.functions import col, asc, desc, sum, avg, max, min, count, countDistinct, round, when, lit, log1p, sqrt, cbrt, log
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline
from sklearn.metrics import confusion_matrix

spark = SparkSession.builder     .master("local[*]")     .appName("Bank Deposit Classification Mini Project")     .config('spark.sql.execution.arrow.pyspark.enabled', True)     .config('spark.sql.session.timeZone', 'UTC')     .config('spark.driver.memory', '32G')     .config('spark.executor.memory', '4g')     .config('spark.ui.showConsoleProgress', True)     .config('spark.sql.repl.eagerEval.enabled', True)     .config('spark.executor.extraJavaOptions', '-XX:+UseG1GC')     .config('spark.driver.extraJavaOptions', '-XX:+UseG1GC')     .config('spark.network.timeout', '800s')     .config('spark.executor.heartbeatInterval', '60s')     .getOrCreate()

sc = spark.sparkContext
sqlContext = SQLContext(sc)


# In[2]:


from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

schema = StructType([
    StructField("Age", IntegerType()),
    StructField("Job", StringType()),
    StructField("Marital", StringType()),
    StructField("Education", StringType()),
    StructField("Default", StringType()),
    StructField("Housing", StringType()),
    StructField("Loan", StringType()),
    StructField("Contact", StringType()),
    StructField("Month", StringType()),
    StructField("Day_of_week", StringType()),
    StructField("Duration", IntegerType()),
    StructField("Campaign", IntegerType()),
    StructField("Pdays", IntegerType()),
    StructField("Previous", IntegerType()),
    StructField("Poutcome", StringType()),
    StructField("Emp_var_rate", DoubleType()),
    StructField("Cons_price_idx", DoubleType()),
    StructField("Cons_conf_idx", DoubleType()),
    StructField("Euribor3m", DoubleType()),
    StructField("Nr_employed", DoubleType()),
    StructField("y", StringType())
])

df = spark.read.csv("XYZ_Bank_Deposit_Data_Classification.csv", header=True, schema=schema, sep=";")
df.show()


# In[3]:


df.printSchema()


# # EDA

# ## Basic Understanding

# In[4]:


df.describe().show()


# ## Check for Missing Values

# In[5]:


from pyspark.sql.functions import col, sum
df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns)).show()


# ## Univariate Analysis

# In[6]:


# Numerical Variables
numerical_vars = [c for c, dtype in df.dtypes if dtype in ['int', 'double']]
df.select(numerical_vars).describe().show()


# In[45]:


# Convert Spark DataFrame to Pandas DataFrame for visualization
# Limiting the data for visualization
pandas_df = df.toPandas()  

# List of numerical columns
numerical_vars = [c for c, dtype in df.dtypes if dtype in ['int', 'double']]

# Plotting histograms for each numerical column
for col in numerical_vars:
    plt.figure(figsize=(10, 6))
    sns.histplot(pandas_df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f'EDA//univariate//{col}.png', bbox_inches='tight')


# In[46]:


# Pdays are slightly different because of 999 that means not contacted before
df.select('Pdays').distinct().show()

# Convert Spark DataFrame to Pandas DataFrame for visualization
# Limiting the data for visualization
pandas_df = df.toPandas()  

plt.figure(figsize=(10, 6))
sns.histplot(pandas_df['Pdays'], kde=True, bins=200)
plt.xlim(0,40)
plt.ylim(0,1000)
plt.title(f'Distribution of Pdays')
plt.xlabel(col)
plt.ylabel('Frequency')
plt.savefig(f'EDA//univariate//Pdays.png', bbox_inches='tight')


# In[20]:


# bin number of employees Less than 5000, 5000-5100 and more than 5100
def helper_function_nbr_employees_bin(y):
    if y < 5000:
        return "Less than 5000"
    elif y >= 5000 and y < 5100:
        return "5000-5100"
    else:
        return "More than 5100"

udf_nr_employed = udf(helper_function_nbr_employees_bin, StringType())
df = df.withColumn("Nbr_Employees_Bin", udf_nr_employed('Nr_employed'))


pDays_target_bivariate_analysis = df.groupBy('Nbr_Employees_Bin').agg(
    (F.sum(F.when(F.col('y') == 'yes', 1).otherwise(0)) / F.count(F.col('y'))).alias('y_yes_perc'),
    (F.avg(F.col('Pdays'))).alias('avg_pdays')
).toPandas()

plt.figure(figsize=(10, 6))
sns.barplot(x='Nbr_Employees_Bin', y='y_yes_perc', data=pDays_target_bivariate_analysis)
plt.title('Percentage of Yes in Target Variable (Conversion Rate) for each Bin in Nr_Employed')
plt.xlabel('Nbr_Employees_Bin')
plt.ylabel('Percentage of Yes in Target Variable (Conversion Rate)')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Nbr_Employees_Bin', y='avg_pdays', data=pDays_target_bivariate_analysis)
plt.title('Average Pdays for each Bin in Nr_Employed')
plt.xlabel('Nbr_Employees_Bin')
plt.ylabel('Average Pdays')
plt.show()


# In[47]:


# Categorical Variables
categorical_vars = [c for c, dtype in df.dtypes if dtype == 'string']
for col in categorical_vars:
    df.groupBy(col).count().orderBy("count", ascending=False).show()

# Plot the distribution of the categorical variables
for col in categorical_vars:
    df.groupBy(col).count().orderBy("count", ascending=False).toPandas().plot.bar(col)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f'EDA//univariate//{col}.png', bbox_inches='tight')


# ## Bivariate Analysis

# In[48]:


for col in categorical_vars:
    df.groupBy(col, 'y').count().show()


# In[49]:


from pyspark.sql.functions import col

result_dfs = []
for col_name in categorical_vars:
    if col_name != "y":
        # Count the occurrences of each category
        count_df = df.groupBy(col_name, 'y').count()

        # Calculate the total count for each category
        total_count_df = df.groupBy(col_name).count().withColumnRenamed("count", "total_count")

        # Join the count with the total count
        percentage_df = count_df.join(total_count_df, col_name)

        # Calculate the percentage
        percentage_df = percentage_df.withColumn("percentage", col("count") / col("total_count") * 100)

        # Keep the dataframe for later plotting
        result_dfs.append(percentage_df)


# In[50]:


result_dfs


# In[51]:


import pandas as pd
import matplotlib.pyplot as plt

for percentage_df in result_dfs:
    # Convert to Pandas DataFrame
    pandas_df = percentage_df.toPandas()

    # Pivot for better plotting
    pivot_df = pandas_df.pivot(index=percentage_df.columns[0], columns='y', values='percentage')

    # Sort by percentage of 'yes' in descending order
    pivot_df = pivot_df.sort_values(by='yes', ascending=False)

    # Plotting
    pivot_df.plot(kind='bar', stacked=True)
    plt.ylabel('Percentage')
    plt.title(f'Percentage Distribution in {percentage_df.columns[0]} by y')
    plt.savefig(f'EDA//bivariate//{percentage_df.columns[0]}.png', bbox_inches='tight')
    plt.show()


# ## Correlation Analysis

# In[52]:


df = df.drop('label')


# In[53]:


# label encode y variable and create correlation matrix for numerical variables including label encoded y variable
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

# label encode y variable
indexer = StringIndexer(inputCol="y", outputCol="label")
df = indexer.fit(df).transform(df)


# In[54]:


numeric_features = [t[0] for t in df.dtypes if t[1] != 'string']
numeric_features_df=df.select(numeric_features)
numeric_features_df.toPandas().head()
col_names =numeric_features_df.columns
features = numeric_features_df.rdd.map(lambda row: row[0:])
corr_mat=Statistics.corr(features, method="pearson")
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = col_names, col_names

corr_df


# In[55]:


# plot correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.savefig(f'EDA//correlation_matrix.png', bbox_inches='tight')
plt.show()


# In[56]:


# plot correlation bar plot to label
corr_df['label'].sort_values(ascending=False).drop('label').plot.bar()
plt.title("Correlation with Target Variable")
plt.ylabel("Correlation Coefficient")
plt.savefig(f'EDA//correlation_barplot.png', bbox_inches='tight')


# ## Transformation Check
# - binary encoding: pdays (999 means no contact)
# - log+1: campaign, duration

# In[57]:


# plot transformed age with bins
df.select('age').toPandas().hist(bins=20)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# In[60]:


# plot campaign with bins
df.select('campaign').toPandas().hist(bins=20)
plt.title("Distribution of Campaign")
plt.xlabel("Campaign")
plt.ylabel("Frequency")
plt.show()


# In[73]:



# Assuming 'df' is your Spark DataFrame and 'campaign' is the column you wish to transform.
# Apply logarithmic transformation to 'campaign'.
df = df.withColumn("campaign_log", log1p("campaign"))

# Now, you would plot the distribution of the transformed 'campaign' to evaluate the result.
# The following code would be executed in a PySpark environment that supports plotting, like Databricks.
# For local environments, you would collect the data to the driver node and use a local plotting library like matplotlib.

# Collect the data to the driver node. Be cautious with large datasets as this can cause out-of-memory errors.
campaign_log_data = df.select('campaign_log').rdd.flatMap(lambda x: x).collect()

# Use matplotlib to plot the histogram.
import matplotlib.pyplot as plt

plt.hist(campaign_log_data, bins=20)
plt.title('Distribution of Log-transformed Campaign')
plt.xlabel('Log of Campaign')
plt.ylabel('Frequency')
plt.show()


# In[74]:


# plot campaign with bins
df.select('duration').toPandas().hist(bins=20)
plt.title("Distribution of Duration")
plt.xlabel("Duration")
plt.ylabel("Frequency")
plt.show()


# In[75]:


# Assuming 'df' is your Spark DataFrame and 'campaign' is the column you wish to transform.
# Apply logarithmic transformation to 'campaign'.
df = df.withColumn("duration_log", log1p("duration"))

# Now, you would plot the distribution of the transformed 'campaign' to evaluate the result.
# The following code would be executed in a PySpark environment that supports plotting, like Databricks.
# For local environments, you would collect the data to the driver node and use a local plotting library like matplotlib.

# Collect the data to the driver node. Be cautious with large datasets as this can cause out-of-memory errors.
duration_log_data = df.select('duration_log').rdd.flatMap(lambda x: x).collect()

# Use matplotlib to plot the histogram.
import matplotlib.pyplot as plt

plt.hist(duration_log_data, bins=20)
plt.title('Distribution of Log-transformed Duration')
plt.xlabel('Log of Duration')
plt.ylabel('Frequency')
plt.show()


# In[76]:


# plot campaign with bins
df.select('nr_employed').toPandas().hist(bins=20)
plt.title("Distribution of nr_employed")
plt.xlabel("nr_employed")
plt.ylabel("Frequency")
plt.show()


# In[78]:


# Assuming 'df' is your Spark DataFrame and 'campaign' is the column you wish to transform.
# Apply logarithmic transformation to 'campaign'.
df = df.withColumn("nr_employed_cbrt", cbrt("nr_employed"))

# Now, you would plot the distribution of the transformed 'campaign' to evaluate the result.
# The following code would be executed in a PySpark environment that supports plotting, like Databricks.
# For local environments, you would collect the data to the driver node and use a local plotting library like matplotlib.

# Collect the data to the driver node. Be cautious with large datasets as this can cause out-of-memory errors.
nr_employed_cbrt_data = df.select('nr_employed_cbrt').rdd.flatMap(lambda x: x).collect()

# Use matplotlib to plot the histogram.
import matplotlib.pyplot as plt

plt.hist(nr_employed_cbrt_data, bins=20)
plt.title('Distribution of Cbrt-transformed nr_employed')
plt.xlabel('Cbrt of nr_employed')
plt.ylabel('Frequency')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




