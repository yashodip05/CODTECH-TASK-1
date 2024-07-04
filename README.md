# CODTECH-TASK-1

NAME:YASHODIP KAMBLE
Company:CODTECH IT SOLUTIONS
ID: CT6DA543
DOMAIN:DATA ANALYTICS
Duration:June to August 2024


OVERVIEW OF PROJECT:

PROJECT:Exploratory Data Analysis (EDA) on Titanic Dataset

Objective:
The goal of this project is to perform an Exploratory Data Analysis (EDA) on the Titanic dataset using Python libraries such as pandas, numpy, matplotlib, and seaborn. The EDA process involves examining the dataset to uncover patterns, relationships, and anomalies, and summarizing the main characteristics of the data.
Steps in the EDA Process:

    Data Collection and Loading:
        Load the Titanic dataset using pandas.

    Initial Data Inspection:
        Display the first few rows of the dataset.
        Check for data types, missing values, and basic statistics.

    Data Cleaning:
        Handle missing values (e.g., imputation, removal).
        Correct data types if necessary.
        Remove duplicates if present.

    Descriptive Statistics:
        Calculate and display summary statistics (mean, median, mode, standard deviation).
        Identify any outliers using statistical methods or visualization.

    Univariate Analysis:
        Plot histograms and box plots to understand the distribution of individual variables.
        Identify any skewness or kurtosis.

    Bivariate Analysis:
        Use scatter plots to explore relationships between pairs of variables.
        Use correlation matrices and heatmaps to identify potential correlations.

    Multivariate Analysis:
        Explore interactions between multiple variables using pair plots or 3D plots.

    Data Visualization:
        Use matplotlib and seaborn to create informative visualizations.
        Include histograms, scatter plots, box plots, and heatmaps to communicate findings.

    Summary and Insights:
        Summarize the main findings from the EDA.
        Highlight any interesting patterns, correlations, or anomalies.
        Provide a preliminary understanding of the dataset's structure and relationships.

Implementation

Here is the code to perform EDA on the Titanic dataset:

python

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
titanic_df = pd.read_csv('titanic.csv')

# Initial data inspection
print(titanic_df.head())
print(titanic_df.info())
print(titanic_df.describe())

# Handling missing values
# Check for missing values
print(titanic_df.isnull().sum())

# Fill missing 'Age' values with median
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with mode
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' column due to too many missing values
titanic_df.drop(columns=['Cabin'], inplace=True)

# Descriptive statistics after cleaning
print(titanic_df.describe())

# Univariate Analysis
plt.figure(figsize=(10, 6))
sns.histplot(titanic_df['Age'], kde=True)
plt.title('Distribution of Age')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Fare', data=titanic_df)
plt.title('Boxplot of Fare by Pclass')
plt.show()

# Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=titanic_df)
plt.title('Age vs Fare')
plt.show()

# Correlation Matrix and Heatmap
corr_matrix = titanic_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Summary and Insights
summary = """
- The Titanic dataset contains 891 samples with 12 features each after dropping the 'Cabin' column.
- The distribution of Age is roughly normal, with some missing values imputed by the median.
- The boxplot shows that higher class passengers (Pclass 1) generally paid higher fares.
- There is a slight negative correlation between Age and Fare.
- The heatmap reveals that Pclass and Fare have a moderate negative correlation, indi
cating that higher class tickets were more expensive.
"""
print(summary)

Summary and Insights

    Data Distribution: The Titanic dataset contains 891 samples with 12 features each after dropping the 'Cabin' column. The distribution of Age is roughly normal, with some missing values imputed by the median.
    Class and Fare: The boxplot shows that higher class passengers (Pclass 1) generally paid higher fares.
    Relationships: There is a slight negative correlation between Age and Fare.
    Correlations: The heatmap reveals that Pclass and Fare have a moderate negative correlation, indicating that higher class tickets were more expensive.

This EDA provides a solid understanding of the Titanic dataset's structure, relationships, and potential areas for further analysis, such as survival prediction modeling.
![Screenshot from 2024-07-04 23-22-02](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/1074234c-3b77-4566-bc62-261d1dc694b4)
![Screenshot from 2024-07-04 23-22-13](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/f3a57ea3-0b21-416d-9d3e-701dac0d9943)
![Screenshot from 2024-07-04 23-22-19](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/39453e72-638a-4033-a63f-a18da8acab2e)
![Screenshot from 2024-07-04 23-22-26](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/2989acf9-73a4-4b42-938c-387d550b9b1e)
![Screenshot from 2024-07-04 23-22-48](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/65651a89-74fa-4ead-8a1c-79c92d9ae7ff)
![Screenshot from 2024-07-04 23-22-57](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/9b0cee82-baf7-486c-b0f5-4a2889005260)
![Screenshot from 2024-07-04 23-22-34](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/9671f790-3966-4d51-86cb-b980adfc227d)
![Screenshot from 2024-07-04 23-23-19](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/07c91fc8-5e0d-4990-ae44-77a0647b4030)
![Screenshot from 2024-07-04 23-23-04](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/d1fa5f6c-8c66-4e89-a183-37317b57bbfb)
![Screenshot from 2024-07-04 23-23-29](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/37e7e5c6-2be6-4d84-b726-ab07fccf92a7)
![Screenshot from 2024-07-04 23-23-33](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/415d9a1d-babe-406a-8943-f0dbe08f8236)
![Screenshot from 2024-07-04 23-23-38](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/99b58733-11a2-4958-a5b3-4101ca8a320e)
![Screenshot from 2024-07-04 23-23-42](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/fe1668f6-accd-40be-a8f6-ca3a96cd3666)
![Screenshot from 2024-07-04 23-23-48](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/39d130d3-de27-45db-9783-476cd5ad0e99)
![Screenshot from 2024-07-04 23-23-55](https://github.com/yashodip05/CODTECH-TASK-1/assets/132188351/554eb2e6-38c6-4915-a7ff-6e291fe944e0)
