# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 17:05:43 2023

@author: Belma
"""
#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

#Import Dataset 
kick_df = pd.read_csv("C:\\Users\\Belma\\Downloads\\KickStarter.csv")

#########################
###Data Pre-Processing###
#########################-

#Checking missing values
print(kick_df.isnull().sum())
# Drop missing values
kick_df = kick_df.dropna()

# Checking duplicates
kick_df[kick_df.duplicated()].shape

# Drop duplicates
kick_df = kick_df.drop_duplicates()
kick_df.shape

# Checking data types
kick_df.dtypes
kick_df.info()

# Drop the predictors that are not there when we lauch the project 
kick_df = kick_df.drop(columns=['state_changed_at_month', 'state_changed_at_day'
                                ,'state_changed_at_yr', 'state_changed_at_hr',
                                'state_changed_at' ,'state_changed_at_weekday',
                                'pledged', 'spotlight', 'backers_count',
                                'launch_to_state_change_days', 'usd_pledged'], axis=1)

# View different values of target variable 
unique_values = kick_df['state'].unique()
print(unique_values)

# Remove observations with state = canceled or suspended 
value_to_remove = 'canceled'
kick_df.drop(kick_df[kick_df['state'] == value_to_remove].index, inplace=True)
value_to_remove2 = 'suspended'
kick_df.drop(kick_df[kick_df['state'] == value_to_remove2].index, inplace=True)

#############
###Country###
#############

# Number of Countries
unique_countries = kick_df['country'].nunique()

print(f"The number of unique countries in the dataset is: {unique_countries}")

# List of country names
unique_countries = kick_df['country'].unique()

print("Unique countries in the dataset:")
for country in unique_countries:
    print(country)

# Dictionary mapping countries to continents
country_to_continent = {
    'GB': 'Europe',
    'US': 'North America',
    'AU': 'Australia (Oceania)',
    'CA': 'North America',
    'NO': 'Europe',
    'FR': 'Europe',
    'BE': 'Europe',
    'NZ': 'Australia (Oceania)',
    'IT': 'Europe',
    'SE': 'Europe',
    'IE': 'Europe',
    'DK': 'Europe',
    'ES': 'Europe',
    'DE': 'Europe',
    'NL': 'Europe',
    'CH': 'Europe',
    'AT': 'Europe',
    'LU': 'Europe'
}

# Dictionary mapping continents to numerical values
continent_to_number = {
    'Europe': 1,
    'North America': 2,
    'Australia (Oceania)': 3
    # Add more continents and their numerical values as needed
}

# Assuming 'kick_df' has a column 'Country' containing country codes
# Add 'Continent' column to 'kick_df' based on the country code
kick_df['Continent'] = kick_df['country'].map(country_to_continent)

# Map continent names to numerical codes
kick_df['Continent_Code'] = kick_df['Continent'].map(continent_to_number)

# Display the updated DataFrame
print(kick_df)

# Drop 'Continent' and 'Country' columns 
kick_df.drop(columns=['Continent', 'country'], inplace=True)

# Display the updated DataFrame
print(kick_df)

################
###Staff Pick###
################

kick_df = kick_df.drop('staff_pick', axis=1)

############################
###Disbaled Communication###
############################

# Count the number of unique values in the 'disabled_communication' column
unique_values_count = kick_df['disable_communication'].nunique()

print("Number of different values in 'disable_communication' column:", unique_values_count)

# Drop this column since it is the same value 
kick_df = kick_df.drop('disable_communication', axis=1)

#######################
###Weekday Variables###
#######################

# Dictionary mapping weekdays to numerical values
weekday_to_number = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

# Replace deadline_weekdays with corresponding numbers
kick_df['deadline_weekday'] = kick_df['deadline_weekday'].map(weekday_to_number)

# Replace created at_weekday with corresponding numbers
kick_df['created_at_weekday'] = kick_df['created_at_weekday'].map(weekday_to_number)

# Replace launched at weekday with corresponding numbers
kick_df['launched_at_weekday'] = kick_df['launched_at_weekday'].map(weekday_to_number)

########################
###Category Variables###
########################

# Number of Categories
unique_categories = kick_df['category'].nunique()

print(f"The number of unique categories in the dataset is: {unique_categories}")

# List of categorie names
unique_categories = kick_df['category'].unique()

print("Unique categories in the dataset:")
for category in unique_categories:
    print(category)
    
# Dictionary with categories as keys and respective project types as values
buckets = {
    'Arts & Entertainment': ['Plays','Festivals', 'Experimental','Musical','Immersive','Sound','Thrillers','Webseries','Blues','Shorts'],
    'Design & Tech': ['Gadgets','Spaces','Web','Apps','Wearables','Software','Hardware','Robots','Makerspaces',
                      'Flight', 'Places'],
    'Academics': ['Academic']
}

# Create a function to assign categories to project types
def categorize_project_type(project):
    for category, types in buckets.items():
        if project in types:
            return category
    return 'Other'  # Assign 'Other' if the project type doesn't match any category

# Create a new column 'category' based on the categorization
kick_df['category_bucket'] = kick_df['category'].apply(categorize_project_type)

# Using Pandas' get_dummies function for creating dummy variables
dummy_variables = pd.get_dummies(kick_df['category_bucket'], prefix='category')

# Concatenating the dummy variables with the original DataFrame
kick_df = pd.concat([kick_df, dummy_variables], axis=1)

# Drop the Category Column
kick_df = kick_df.drop('category', axis=1)

# Drop the bucket category column
kick_df = kick_df.drop('category_bucket', axis=1)

##############
###Currency###
##############

kick_df = kick_df.drop('currency', axis=1)

##############
###Goal USD###
##############

# Create a new column 'goal_usd' representing the funding goal in USD
kick_df['goal_usd'] = kick_df['goal'] * kick_df['static_usd_rate']

kick_df = kick_df.drop('goal', axis=1)


#################
###Launched at###
#################

kick_df = kick_df.drop('launched_at', axis=1)

################
###Created at###
################

kick_df = kick_df.drop('created_at', axis=1)


##############
###Deadline###
##############

kick_df = kick_df.drop('deadline', axis=1)


##########
###Name###
##########

# Drop the name column
kick_df = kick_df.drop('name', axis=1)

########
###ID###
########

# Drop the ID column
kick_df = kick_df.drop('id', axis=1)

###########
###State###
###########

# Convert 'state' column to binary
kick_df['state'] = (kick_df['state'] == 'successful').astype(int)

# Display the updated DataFrame
print(kick_df['state'])


##########################
###Name Len & Blurb Len###
##########################

# Drop the Name Len column
kick_df = kick_df.drop('name_len', axis=1)

# Drop the Blrub Len column
kick_df = kick_df.drop('blurb_len', axis=1)


#############################
###Drop Temporal Variables###
#############################

# Drop the Name Len column
kick_df = kick_df.drop('deadline_weekday', axis=1)

# Drop the Name Len column
kick_df = kick_df.drop('created_at_weekday', axis=1)

# Drop the Name Len column
kick_df = kick_df.drop('created_at_day', axis=1)

# Drop the Name Len column
kick_df = kick_df.drop('created_at_month', axis=1)

# Drop the Name Len column
kick_df = kick_df.drop('created_at_hr', axis=1)

# Drop the Name Len column
kick_df = kick_df.drop('launched_at_weekday', axis=1)

# Drop the Name Len column
kick_df = kick_df.drop('launched_at_month', axis=1)

# Drop the Name Len column
kick_df = kick_df.drop('launched_at_day', axis=1)

# Drop the Name Len column
kick_df = kick_df.drop('launched_at_hr', axis=1)

# Drop the Name Len column
kick_df = kick_df.drop('deadline_day', axis=1)

# Drop the Name Len column
kick_df = kick_df.drop('deadline_month', axis=1)

# Drop the Name Len column
kick_df = kick_df.drop('deadline_hr', axis=1)

########################
###Correlation Matrix###
########################

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = kick_df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(20, 15))

# Adjust font size for annotations
annot_kws = {'fontsize': 10}  
 
# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Set the title of the plot
plt.title('Correlation Matrix Heatmap')

# Display the plot
plt.show()

# Select upper triangle of correlation matrix
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

# Drop created at
kick_df = kick_df.drop('created_at_yr', axis=1)

#######################
###Handling Outliers###
#######################

import pandas as pd
from scipy import stats

# Assuming 'df' is your DataFrame and 'column_name' is the column you want to analyze
column_name = 'goal'

# Calculate Z-Score for the specified column
z_scores = stats.zscore(kick_df["goal_usd"])

# Define a threshold for outlier detection (commonly Z-score > 3 or < -3)
threshold = 3

# Identify outliers
outliers = kick_df[abs(z_scores) > threshold]

# Display the outliers
print(outliers)
  
# Drop the rows with outliers
kick_df = kick_df.drop(outliers.index)

# Display the cleaned DataFrame
print(kick_df)

#======================================================================================================

##############################
###CLASSIFICATION ALGORITHM###
##############################

# Split data into train and test sets
X = kick_df.drop(columns=['state'],  axis=1)
y = kick_df['state']

# Create polynomial features including interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

########################################################################
###Gradient Boosting with Hyperparameter Tuning and Feature Selection###
########################################################################

# Assume X and y are your Pandas DataFrame for feature and target variables

# Convert Pandas DataFrames to NumPy arrays
X = X.values
y = y.values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting classifier
gb = GradientBoostingClassifier(random_state=42)

# Fit the classifier on the training data
gb.fit(X_train, y_train)

# Get feature importances and sort them
feature_importances = gb.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]  # Sort indices in descending order of importance

# Select top k features based on importance score (e.g., top 10 features)
k = 10
selected_features = sorted_indices[:k]

# Use only the selected top k features
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

# Create a new Gradient Boosting classifier with the selected features
gb_selected = GradientBoostingClassifier(random_state=42)

# Perform GridSearchCV with cross-validation on the new model with selected features
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(estimator=gb_selected, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_selected, y_train)

# Get the best model and its parameters
best_gb_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate the best model on the test set using selected features
y_pred_best = best_gb_model.predict(X_test_selected)
accuracy_best = accuracy_score(y_test, y_pred_best)

print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy for Best Gradient Boosting Model with Selected Features: {accuracy_best}")

# Train the Gradient Boosting model with k-fold cross-validation
gbm_cv_scores = cross_val_score(gb, X_train, y_train, cv=10, scoring='accuracy')
print(f'Gradient Boosting Cross-Validation Scores: {gbm_cv_scores}')
print(f'Mean Accuracy: {gbm_cv_scores.mean()}')

#======================================================================================================

##########################
###CLUSTERING ALGORITHM###
##########################

######################################
###Silhouette Score - K-Means + PCA###
######################################

#Import Libraries
from sklearn.metrics import silhouette_score

scaler = StandardScaler()
features = kick_df.drop(['state'], axis=1)  # Assuming 'state' is your target variable
features_scaled = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Silhouette Analysis on PCA Features with K-Means
silhouette_scores_pca_kmeans = []
wcss_pca_kmeans = []  # For the Elbow Method
for i in range(2, 20):
    kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_pca.fit(features_pca)
    silhouette_avg_pca = silhouette_score(features_pca, kmeans_pca.labels_)
    silhouette_scores_pca_kmeans.append(silhouette_avg_pca)
    # Elbow Method
    wcss_pca_kmeans.append(kmeans_pca.inertia_)
    
    # Output silhouette score for each K-Means model with PCA features
    print(f"Silhouette Score for K-Means with {i} clusters on PCA Features: {silhouette_avg_pca}")

# Plotting silhouette scores for K-Means on PCA Features
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(range(2, 20), silhouette_scores_pca_kmeans, label='PCA + K-Means')
plt.title('Silhouette Scores - PCA + K-Means')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.legend()

# Elbow Method Plot
plt.subplot(2, 1, 2)
plt.plot(range(2, 20), wcss_pca_kmeans, marker='o')
plt.title('Elbow Method for PCA + K-Means')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')

plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=15, random_state=42)
kmeans_clusters = kmeans.fit_predict(features_pca)

def plot_clusters(clusters, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis', marker='o')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()
    plt.show()

plot_clusters(kmeans_clusters, "K-Means Clusters")

#=============================================================================

###############################################################################
#############################GRADING###########################################
###############################################################################

#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

#########################
###Import Grading Data###
#########################

kickstarter_grading_df = pd.read_excel("C:\\Users\\Belma\\Downloads\\Kickstarter-Grading-Sample (2).xlsx")

##############################
###Pre-Process Grading Data###
##############################

#Checking missing values
print(kickstarter_grading_df.isnull().sum())
# Drop missing values
kickstarter_grading_df = kickstarter_grading_df.dropna()

# Checking duplicates
kickstarter_grading_df[kickstarter_grading_df.duplicated()].shape

# Drop duplicates
kickstarter_grading_df = kickstarter_grading_df.drop_duplicates()
kickstarter_grading_df.shape

# Checking data types
kickstarter_grading_df.dtypes
kickstarter_grading_df.info()

# Drop the predictors that are not there when we lauch the project 
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['state_changed_at_month', 'state_changed_at_day'
                                ,'state_changed_at_yr', 'state_changed_at_hr',
                                'state_changed_at' ,'state_changed_at_weekday',
                                'pledged', 'spotlight', 'backers_count',
                                'launch_to_state_change_days', 'usd_pledged'], axis=1)

# View different values of target variable 
unique_values = kickstarter_grading_df['state'].unique()
print(unique_values)

# Remove observations with state = canceled or suspended 
value_to_remove = 'canceled'
kickstarter_grading_df.drop(kickstarter_grading_df[kickstarter_grading_df['state'] == value_to_remove].index, inplace=True)
value_to_remove2 = 'suspended'
kickstarter_grading_df.drop(kickstarter_grading_df[kickstarter_grading_df['state'] == value_to_remove2].index, inplace=True)

#############
###Country###
#############

# Number of Countries
unique_countries = kickstarter_grading_df['country'].nunique()

print(f"The number of unique countries in the dataset is: {unique_countries}")

# List of country names
unique_countries = kickstarter_grading_df['country'].unique()

print("Unique countries in the dataset:")
for country in unique_countries:
    print(country)

# Dictionary mapping countries to continents
country_to_continent = {
    'GB': 'Europe',
    'US': 'North America',
    'AU': 'Australia (Oceania)',
    'CA': 'North America',
    'NO': 'Europe',
    'FR': 'Europe',
    'BE': 'Europe',
    'NZ': 'Australia (Oceania)',
    'IT': 'Europe',
    'SE': 'Europe',
    'IE': 'Europe',
    'DK': 'Europe',
    'ES': 'Europe',
    'DE': 'Europe',
    'NL': 'Europe',
    'CH': 'Europe',
    'AT': 'Europe',
    'LU': 'Europe'
}

# Dictionary mapping continents to numerical values
continent_to_number = {
    'Europe': 1,
    'North America': 2,
    'Australia (Oceania)': 3
    # Add more continents and their numerical values as needed
}

# Assuming 'kick_df' has a column 'Country' containing country codes
# Add 'Continent' column to 'kick_df' based on the country code
kickstarter_grading_df['Continent'] = kickstarter_grading_df['country'].map(country_to_continent)

# Map continent names to numerical codes
kickstarter_grading_df['Continent_Code'] = kickstarter_grading_df['Continent'].map(continent_to_number)

# Display the updated DataFrame
print(kickstarter_grading_df)

# Drop 'Continent' and 'Country' columns 
kickstarter_grading_df.drop(columns=['Continent', 'country'], inplace=True)

# Display the updated DataFrame
print(kickstarter_grading_df)

################
###Staff Pick###
################

kickstarter_grading_df = kickstarter_grading_df.drop('staff_pick', axis=1)

############################
###Disbaled Communication###
############################

# Count the number of unique values in the 'disabled_communication' column
unique_values_count = kickstarter_grading_df['disable_communication'].nunique()

print("Number of different values in 'disable_communication' column:", unique_values_count)

# Drop this column since it is the same value 
kickstarter_grading_df = kickstarter_grading_df.drop('disable_communication', axis=1)

#######################
###Weekday Variables###
#######################

# Dictionary mapping weekdays to numerical values
weekday_to_number = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

# Replace deadline_weekdays with corresponding numbers
kickstarter_grading_df['deadline_weekday'] = kickstarter_grading_df['deadline_weekday'].map(weekday_to_number)

# Replace created at_weekday with corresponding numbers
kickstarter_grading_df['created_at_weekday'] = kickstarter_grading_df['created_at_weekday'].map(weekday_to_number)

# Replace launched at weekday with corresponding numbers
kickstarter_grading_df['launched_at_weekday'] = kickstarter_grading_df['launched_at_weekday'].map(weekday_to_number)

########################
###Category Variables###
########################

# Number of Categories
unique_categories = kickstarter_grading_df['category'].nunique()

print(f"The number of unique categories in the dataset is: {unique_categories}")

# List of categorie names
unique_categories = kickstarter_grading_df['category'].unique()

print("Unique categories in the dataset:")
for category in unique_categories:
    print(category)
    
# Dictionary with categories as keys and respective project types as values
buckets = {
    'Arts & Entertainment': ['Plays','Festivals', 'Experimental','Musical','Immersive','Sound','Thrillers','Webseries','Blues','Shorts'],
    'Design & Tech': ['Gadgets','Spaces','Web','Apps','Wearables','Software','Hardware','Robots','Makerspaces',
                      'Flight', 'Places'],
    'Academics': ['Academic']
}

# Create a function to assign categories to project types
def categorize_project_type(project):
    for category, types in buckets.items():
        if project in types:
            return category
    return 'Other'  # Assign 'Other' if the project type doesn't match any category

# Create a new column 'category' based on the categorization
kickstarter_grading_df['category_bucket'] = kickstarter_grading_df['category'].apply(categorize_project_type)

# Using Pandas' get_dummies function for creating dummy variables
dummy_variables = pd.get_dummies(kickstarter_grading_df['category_bucket'], prefix='category')

# Concatenating the dummy variables with the original DataFrame
kickstarter_grading_df = pd.concat([kickstarter_grading_df, dummy_variables], axis=1)

# Drop the Category Column
kickstarter_grading_df = kickstarter_grading_df.drop('category', axis=1)

# Drop the bucket category column
kickstarter_grading_df = kickstarter_grading_df.drop('category_bucket', axis=1)

##############
###Currency###
##############

kickstarter_grading_df = kickstarter_grading_df.drop('currency', axis=1)

##############
###Goal USD###
##############

# Create a new column 'goal_usd' representing the funding goal in USD
kickstarter_grading_df['goal_usd'] = kickstarter_grading_df['goal'] * kickstarter_grading_df['static_usd_rate']

kickstarter_grading_df = kickstarter_grading_df.drop('goal', axis=1)


#################
###Launched at###
#################

kickstarter_grading_df = kickstarter_grading_df.drop('launched_at', axis=1)

################
###Created at###
################

kickstarter_grading_df = kickstarter_grading_df.drop('created_at', axis=1)


##############
###Deadline###
##############

kickstarter_grading_df = kickstarter_grading_df.drop('deadline', axis=1)


##########
###Name###
##########

# Drop the name column
kickstarter_grading_df = kickstarter_grading_df.drop('name', axis=1)

########
###ID###
########

# Drop the ID column
kickstarter_grading_df = kickstarter_grading_df.drop('id', axis=1)

###########
###State###
###########

# Convert 'state' column to binary
kickstarter_grading_df['state'] = (kickstarter_grading_df['state'] == 'successful').astype(int)

# Display the updated DataFrame
print(kickstarter_grading_df['state'])


##########################
###Name Len & Blurb Len###
##########################

# Drop the Name Len column
kickstarter_grading_df = kickstarter_grading_df.drop('name_len', axis=1)

# Drop the Blrub Len column
kickstarter_grading_df = kickstarter_grading_df.drop('blurb_len', axis=1)


#############################
###Drop Temporal Variables###
#############################

# Drop the Name Len column
kickstarter_grading_df = kickstarter_grading_df.drop('deadline_weekday', axis=1)

# Drop the Name Len column
kickstarter_grading_df = kickstarter_grading_df.drop('created_at_weekday', axis=1)

# Drop the Name Len column
kickstarter_grading_df = kickstarter_grading_df.drop('created_at_day', axis=1)

# Drop the Name Len column
kickstarter_grading_df = kickstarter_grading_df.drop('created_at_month', axis=1)

# Drop the Name Len column
kickstarter_grading_df = kickstarter_grading_df.drop('created_at_hr', axis=1)

# Drop the Name Len column
kickstarter_grading_df = kickstarter_grading_df.drop('launched_at_weekday', axis=1)

# Drop the Name Len column
kickstarter_grading_df = kickstarter_grading_df.drop('launched_at_month', axis=1)

# Drop the Name Len column
kickstarter_grading_df = kickstarter_grading_df.drop('launched_at_day', axis=1)

# Drop the Name Len column
kickstarter_grading_df = kickstarter_grading_df.drop('launched_at_hr', axis=1)

# Drop the Name Len column
kickstarter_grading_df = kickstarter_grading_df.drop('deadline_day', axis=1)

# Drop the Name Len column
kickstarter_grading_df = kickstarter_grading_df.drop('deadline_month', axis=1)

# Drop the Name Len column
kickstarter_grading_df = kickstarter_grading_df.drop('deadline_hr', axis=1)

########################
###Correlation Matrix###
########################

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = kickstarter_grading_df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(20, 15))

# Adjust font size for annotations
annot_kws = {'fontsize': 10}  
 
# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Set the title of the plot
plt.title('Correlation Matrix Heatmap')

# Display the plot
plt.show()

# Select upper triangle of correlation matrix
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

# Drop created at
kickstarter_grading_df = kickstarter_grading_df.drop('created_at_yr', axis=1)

#######################
###Handling Outliers###
#######################

import pandas as pd
from scipy import stats

# Assuming 'df' is your DataFrame and 'column_name' is the column you want to analyze
column_name = 'goal'

# Calculate Z-Score for the specified column
z_scores = stats.zscore(kickstarter_grading_df["goal_usd"])

# Define a threshold for outlier detection (commonly Z-score > 3 or < -3)
threshold = 3

# Identify outliers
outliers = kickstarter_grading_df[abs(z_scores) > threshold]

# Display the outliers
print(outliers)
  
# Drop the rows with outliers
kickstarter_grading_df = kickstarter_grading_df.drop(outliers.index)

# Display the cleaned DataFrame
print(kickstarter_grading_df)


##############################
###CLASSIFICATION ALGORITHM###
##############################

# Split data into target variable and predictor variable
X = kickstarter_grading_df.drop(columns=['state'],  axis=1)
y = kickstarter_grading_df['state']

# Create polynomial features including interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


########################################################################
###Gradient Boosting with Hyperparameter Tuning and Feature Selection###
########################################################################

# Assume X and y are your Pandas DataFrame for feature and target variables

# Convert Pandas DataFrames to NumPy arrays
X = X.values
y = y.values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting classifier
gb = GradientBoostingClassifier(random_state=42)

# Fit the classifier on the training data
gb.fit(X_train, y_train)

# Get feature importances and sort them
feature_importances = gb.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]  # Sort indices in descending order of importance

# Select top k features based on importance score (e.g., top 10 features)
k = 10
selected_features = sorted_indices[:k]

# Use only the selected top k features
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

# Create a new Gradient Boosting classifier with the selected features
gb_selected = GradientBoostingClassifier(random_state=42)

# Perform GridSearchCV with cross-validation on the new model with selected features
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(estimator=gb_selected, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_selected, y_train)

# Get the best model and its parameters
best_gb_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate the best model on the test set using selected features
y_pred_best = best_gb_model.predict(X_test_selected)
accuracy_best = accuracy_score(y_test, y_pred_best)

print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy for Best Gradient Boosting Model with Selected Features: {accuracy_best}")

# Train the Gradient Boosting model with k-fold cross-validation
gbm_cv_scores = cross_val_score(gb, X_train, y_train, cv=10, scoring='accuracy')
print(f'Gradient Boosting Cross-Validation Scores: {gbm_cv_scores}')
print(f'Mean Accuracy: {gbm_cv_scores.mean()}')

##########################
###CLUSTERING ALGORITHM###
##########################

######################################
###Silhouette Score - K-Means + PCA###
######################################

#Import Libraries
from sklearn.metrics import silhouette_score

scaler = StandardScaler()
features = kickstarter_grading_df.drop(['state'], axis=1)  # Assuming 'state' is your target variable
features_scaled = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Silhouette Analysis on PCA Features with K-Means
silhouette_scores_pca_kmeans = []
wcss_pca_kmeans = []  # For the Elbow Method
for i in range(2, 20):
    kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_pca.fit(features_pca)
    silhouette_avg_pca = silhouette_score(features_pca, kmeans_pca.labels_)
    silhouette_scores_pca_kmeans.append(silhouette_avg_pca)
    # Elbow Method
    wcss_pca_kmeans.append(kmeans_pca.inertia_)
    
    # Output silhouette score for each K-Means model with PCA features
    print(f"Silhouette Score for K-Means with {i} clusters on PCA Features: {silhouette_avg_pca}")

# Plotting silhouette scores for K-Means on PCA Features
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(range(2, 20), silhouette_scores_pca_kmeans, label='PCA + K-Means')
plt.title('Silhouette Scores - PCA + K-Means')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.legend()

# Elbow Method Plot
plt.subplot(2, 1, 2)
plt.plot(range(2, 20), wcss_pca_kmeans, marker='o')
plt.title('Elbow Method for PCA + K-Means')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')

plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=15, random_state=42)
kmeans_clusters = kmeans.fit_predict(features_pca)

def plot_clusters(clusters, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis', marker='o')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()
    plt.show()

plot_clusters(kmeans_clusters, "K-Means Clusters")
