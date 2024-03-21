#%%
# # # TODO: The threshold of 0.5 is arbitrary and might need to be adjusted based on your specific dataset and the model you are using. For some models, even moderately correlated features might pose problems, while for others, even higher correlations might not be as concerning.
# # # TODO: As a baseline model we can also use a model that has built-in mechanisms for feature selection (like L1 regularization for linear models). 
# # # TODO: Saga: Not checking missing values, outliers, or other data quality issues, imbalanced dataset. These can also affect the model's performance and should be addressed before or during feature selection.
#%%
from sklearn.model_selection import train_test_split
import onnxruntime as rt
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from sklearn.preprocessing import StandardScaler
# define a XGBoost classifier
import xgboost as xgb
import warnings
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin


warnings.filterwarnings("ignore")  # Ignore runtime warnings
# Temporarily adjust pandas display settings for large DataFrames
pd.set_option('display.max_rows', 100)  # Ensure 100 rows can be displayed
pd.set_option('display.max_columns', None)  # Ensure all columns can be displayed
pd.set_option('display.width', None)  # Automatically adjust display width to terminal size
pd.set_option('display.max_colwidth', None)  # Ensure full width of column content is shown
pd.set_option('display.float_format', '{:.4f}'.format)  # Format the float numbers for better readability
#%% md
# # Data preprocessing and feature selection
# 
# Our data consists of binary data so we only want to calculate the Z-score for non-binary colomns
#%%
# Load the dataset
data = pd.read_csv('data/synth_data_for_training.csv')
#%%
print("Before cleaning:")
print("Missing values per column:")
print("Total missing values:", data.isna().sum().sum())

# Identify non-binary columns
non_binary_columns = [col for col in data.columns if not (np.isin(data[col].unique(), [0, 1]).all() and len(data[col].unique()) == 2)]

# Calculate Z-scores for non-binary columns only
z_scores_non_binary = np.abs(stats.zscore(data[non_binary_columns], nan_policy='omit'))

# Mask to identify rows with outliers in non-binary columns
outlier_mask = (z_scores_non_binary > 3.5).any(axis=1)

# Select a subset of non-binary columns for plotting to avoid large image sizes
plot_columns = non_binary_columns[:5]  # Adjust this number based on your specific needs

# Plot outliers for the selected columns before removing
plt.figure(figsize=(20, 5))
for i, col in enumerate(plot_columns, 1):
    plt.subplot(1, len(plot_columns), i)
    sns.boxplot(y=data[col])
    plt.title(f'Before: {col}')
plt.tight_layout()
plt.show()

# Remove outliers from the dataset using the previously defined full_outlier_mask
data_cleaned = data[~outlier_mask]

print("After cleaning:")
print("Missing values per column:")
print("Total missing values:", data_cleaned.isna().sum().sum())

# Plot outliers for the selected columns after removing
plt.figure(figsize=(20, 5))
for i, col in enumerate(plot_columns, 1):
    plt.subplot(1, len(plot_columns), i)
    sns.boxplot(y=data_cleaned[col])
    plt.title(f'After: {col}')
plt.tight_layout()
plt.show()

# Print the shape of the dataset before and after cleaning
print("Shape before cleaning:", data.shape)
print("Shape after cleaning:", data_cleaned.shape)
#%%
def filter_non_fair_features(df):
    non_fair_keywords = [
        "adres", "woonadres", "verzendadres", "buurt", "wijk", "plaats", "persoon_geslacht_vrouw", "taal", "kind"
        , "ontheffing"
    ]
    # Optionally, define keywords for features you want to ensure are included
    fair_inclusion_keywords = [
        "medische_omstandigheden", "sociaal_maatschappelijke_situatie"
    ]

    # Maak de controle case-insensitive
    non_fair_keywords = [keyword.lower() for keyword in non_fair_keywords]
    fair_inclusion_keywords = [keyword.lower() for keyword in fair_inclusion_keywords]

    # Filter features, ensuring that certain conditions are met for inclusion or exclusion
    fair_features = [feature for feature in df.columns if not any(nfk in feature.lower() for nfk in non_fair_keywords) or any(fik in feature.lower() for fik in fair_inclusion_keywords)]
    
    # Keep list of removed features
    removed_features = [feature for feature in df.columns if feature not in fair_features]
    
    # Retourneer een DataFrame met alleen de FAIR features
    return df[fair_features], removed_features


# Pas de filter toe op de DataFrame
data_reduced, removed_features = filter_non_fair_features(data_cleaned)

# print all kolomn that are removed
print("Removed features:")
for feature in removed_features:
    print(feature)

# # Print alle kolomnamen van de gefilterde DataFrame
print("\n\n\nRemaining features:")
for col in data_reduced.columns:
    print(col)
    
# Print the shape of the dataset before and after filtering
print("\nShape before filtering:", data_cleaned.shape)
print("Shape after filtering:", data_reduced.shape)



class FeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(self, non_fair_keywords):
        self.non_fair_keywords = non_fair_keywords
        self.original_feature_names = None  # Initialize original feature names
        self.selected_feature_names = None 
    
    def fit(self, X, y=None):
        self.original_feature_names = X.columns.tolist()  # Store original feature names
        self.selected_feature_names = self.transform(X).columns.tolist()
        return self
    
    def transform(self, X):
        # Make the keywords case-insensitive
        non_fair_keywords = [keyword.lower() for keyword in self.non_fair_keywords]

        # Filter features based on non-fair keywords
        fair_features = [feature for feature in X.columns if not any(nfk in feature.lower() for nfk in non_fair_keywords)]

        # Keep track of removed features
        self.removed_features = [feature for feature in X.columns if feature not in fair_features]

        # Return DataFrame with only the fair features
        return X[fair_features]

# Define non-fair keywords
non_fair_keywords = [
    "adres", "woonadres", "verzendadres", "buurt", "wijk", "plaats", "persoon_geslacht_vrouw", "taal", "kind", "ontheffing"
]
#%%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Assuming data_reduced_scaled is already your standardized data
# Standardize the Data (you've already done this part)
scaler = StandardScaler()
data_reduced_scaled = scaler.fit_transform(data_reduced)

# Determine the number of components
pca = PCA().fit(data_reduced_scaled)

# Calculate the cumulative sum of explained variance ratio
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot the elbow plot
plt.figure(figsize=(10, 7))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('PCA Elbow Plot')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)

# Optional: Add a threshold line, e.g., 0.95 for 95% explained variance
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

plt.show()
#%%
# Assuming data_reduced_scaled is your standardized data
# Perform PCA to retain 95% of the variance
pca = PCA(n_components=0.95)
data_reduced_pca = pca.fit_transform(data_reduced_scaled)

# Convert the PCA result back into a pandas DataFrame
# Create column names based on the number of selected components
columns = [f'PC{i+1}' for i in range(data_reduced_pca.shape[1])]
data_reduced_df = pd.DataFrame(data_reduced_pca, columns=columns)
#%%
# Check how imbalance the dataset is
data_reduced['checked'].value_counts(normalize=True)
#%%
# Let's specify the features and the target
y = data_cleaned['checked']
X = data_cleaned.drop(['checked'], axis=1)
X = X.astype(np.float32)

# TODO: Instead of a single train-test split, consider using cross-validation to assess model performance more robustly. This approach can help ensure the model's generalizability across different subsets of our data.
# Let's split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
#%%
# TODO: Further explore feature engineering possibilities. Creating new features based on domain knowledge can provide the model with additional insights, potentially improving performance

# Initializing and training the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Getting feature importances
feature_importances = clf.feature_importances_

# Converting feature importances into a more readable format
features = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sorting features by importance
features_sorted = features.sort_values(by='Importance', ascending=False)

# Plotting the top 50 features
plt.figure(figsize=(20, 10))
sns.barplot(x='Importance', y='Feature', data=features_sorted.head(50))
plt.title('Top 50 features')
plt.show()
#%% md
# # Feature scaling and model training
#%%
classifier = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=1,
    use_label_encoder=False,  # To avoid warning
    eval_metric='logloss',  # Evaluation metric to avoid warning
    random_state=0
)
#%%
# Create a pipeline object with our selector and classifier
# NOTE: You can create custom pipeline objects but they must be registered to onnx or it will not recognise them
# Because of this we recommend using the onnx known objects as defined in the documentation
# TODO: The pipeline construction and inclusion of feature scaling via StandardScaler is a good practice, ensuring that your model is not biased by the scale of the features.
pipeline_steps = [
    ('filter_features', FeatureFilter(non_fair_keywords)),  # Filter non-fair features
    ('scaler', StandardScaler()),  # First, scale the features
    ('pca', PCA(n_components=0.95)),  # Then, apply PCA to reduce dimensionality, retaining 95% variance
    ('classification', classifier)  # Finally, use the classifier for prediction
]

pipeline = Pipeline(steps=pipeline_steps)

# Let's train a simple model
pipeline.fit(X_train, y_train)
#%%
# TODO: Our evaluation focuses on accuracy, which is a good starting point. However, for fraud detection, other metrics like Precision, Recall, F1 Score, or even a custom cost function might be more appropriate due to the typically imbalanced nature of fraud data. This helps ensure you're not only capturing the fraud cases accurately but also minimizing false positives which can be costly or disruptive.
# Let's evaluate the model
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Calculate precision
precision = precision_score(y_test, y_pred)
print(f'Precision: {precision:.4f}')

# Calculate recall
recall = recall_score(y_test, y_pred)
print(f'Recall: {recall:.4f}')

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1:.4f}')

# print confusion matrix
from sklearn.metrics import confusion_matrix


# Adjust the classification threshold
threshold = 0.9  # Set this to the new threshold you want to test
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)  # Apply the new threshold to make predictions

# Evaluate the adjusted predictions
accuracy_adjusted = accuracy_score(y_test, y_pred_adjusted)
precision_adjusted = precision_score(y_test, y_pred_adjusted)
recall_adjusted = recall_score(y_test, y_pred_adjusted)
f1_adjusted = f1_score(y_test, y_pred_adjusted)

print(f'Adjusted Accuracy: {accuracy_adjusted:.4f}')
print(f'Adjusted Precision: {precision_adjusted:.4f}')
print(f'Adjusted Recall: {recall_adjusted:.4f}')
print(f'Adjusted F1 Score: {f1_adjusted:.4f}')

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred_adjusted)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#%%
# Let's convert the model to ONNX
onnx_model = convert_sklearn(
    pipeline, initial_types=[('X', FloatTensorType((None, X.shape[1])))],
    target_opset=12)

# Let's check the accuracy of the converted model
sess = rt.InferenceSession(onnx_model.SerializeToString())
y_pred_onnx =  sess.run(None, {'X': X_test.values.astype(np.float32)})

accuracy_onnx_model = accuracy_score(y_test, y_pred_onnx[0])
print('Accuracy of the ONNX model: ', accuracy_onnx_model)
#%%
# Let's save the model
onnx.save(onnx_model, "model/good_model.onnx")

# Let's load the model
new_session = rt.InferenceSession("model/good_model.onnx")

# Let's predict the target
y_pred_onnx2 =  new_session.run(None, {'X': X_test.values.astype(np.float32)})

accuracy_onnx_model = accuracy_score(y_test, y_pred_onnx2[0])
print('Accuracy of the ONNX model: ', accuracy_onnx_model)

#%%
# Check how imbalance the dataset is
# data_cleaned['checked'].value_counts(normalize=True)
y_train.value_counts(normalize=True)
# y_test.value_counts(normalize=True)
#%%
# Get most important features

selected_features = pipeline.named_steps['filter_features'].selected_feature_names
original_feature_names = pipeline.named_steps['filter_features'].original_feature_names

for feature_name in selected_features:
    print(feature_name)
# original_feature_names = pipeline.named_steps['filter_features'].original_feature_names
# print( original_feature_names)
# # Step 2: Get principal components if using PCA
# if 'pca' in pipeline.named_steps:
#     pca = pipeline.named_steps['pca']
#     # Get the names of the original features
#     original_feature_names = pipeline.named_steps['filter_features'].original_feature_names
#     # Create a list of tuples containing the variance explained by each principal component and its corresponding original features
#     component_variances = [(variance, ', '.join([original_feature_names[i] for i in sorted(range(len(pca.components_[component])), key=lambda x: pca.components_[component][x], reverse=True)])) for component, variance in enumerate(pca.explained_variance_ratio_)]
#     # Sort the list by variance explained
#     component_variances.sort(reverse=True)
#     print("Principal Components and their most important features:")
#     for component, (variance, feature_names) in enumerate(component_variances):
#         print(f"Principal Component {component + 1}: Variance Explained = {variance:.2%}, Features = {feature_names}")

# if 'classification' in pipeline.named_steps:
#     classifier = pipeline.named_steps['classification']
#     if hasattr(classifier, 'feature_importances_'):
#         feature_importances = classifier.feature_importances_
#         # Create a list of tuples containing feature names and their importances
#         importance_tuples = [(feature_name, importance) for feature_name, importance in zip(selected_features, feature_importances)]
#         # Sort the list of tuples by importance
#         sorted_importances = sorted(importance_tuples, key=lambda x: x[1], reverse=True)
#         print("Feature Importances:")
#         for feature_name, importance in sorted_importances:
#             print(f"{feature_name}: {importance:.4f}")
#     elif hasattr(classifier, 'coef_'):
#         coef = classifier.coef_[0]
#         # Create a list of tuples containing feature names and their coefficients
#         coef_tuples = [(feature_name, coef_value) for feature_name, coef_value in zip(selected_features, coef)]
#         # Sort the list of tuples by coefficient absolute value
#         sorted_coefs = sorted(coef_tuples, key=lambda x: abs(x[1]), reverse=True)
#         print("Coefficients:")
#         for feature_name, coef_value in sorted_coefs:
#             print(f"{feature_name}: {coef_value:.4f}")
#     else:
#         print("Feature importances are not available for this classifier.")
#%%
# Testing

first_sample = X_test.iloc[[3]]

# Create a copy of the first sample and flip the boolean value of persoon_geslacht_vrouw
flipped_sample = first_sample.copy()
print(flipped_sample['afspraak_gespr__einde_zoekt___galo_gesprek_'].iloc[0])
print(flipped_sample['persoon_geslacht_vrouw'].iloc[0])
if( flipped_sample['persoon_geslacht_vrouw'].iloc[0] == 0.0 ):
    flipped_sample['persoon_geslacht_vrouw'] = 1.0
else:
    flipped_sample['persoon_geslacht_vrouw'] = 0.0

# Concatenate the original and the modified samples
new_X_test = pd.concat([first_sample, flipped_sample], ignore_index=True)
new_y_pred = pipeline.predict(new_X_test)
print(new_y_pred)
#%%
# Metamorphic testing: Other than fairness testing
# If a value changes then the prediction likelihood should change too in line with the purpose of the model 
# pla_historie_ontwikkeling 0 or 25 // number of developments in PLA history

# Initialize variables to store likelihoods
likelihoods_0 = []
likelihoods_25 = []

# Iterate through each sample in the test set
for index, row in X_test.iterrows():
    # Convert the row to a DataFrame to ensure it's a DataFrame object
    X_sample = pd.DataFrame(row).transpose()

    # Make predictions for 0 developments in PLA history
    X_sample_0 = X_sample.copy()
    X_sample_0['pla_historie_ontwikkeling'] = 0
    y_proba_0 = pipeline.predict_proba(X_sample_0)

    # Make predictions for 25 developments in PLA history
    X_sample_25 = X_sample.copy()
    X_sample_25['pla_historie_ontwikkeling'] = 25
    y_proba_25 = pipeline.predict_proba(X_sample_25)

    # Append the likelihoods for both age groups
    likelihoods_0.append(y_proba_0[:, 1])  # Probability of class 1 (fraud) for 0 developments in PLA history
    likelihoods_25.append(y_proba_25[:, 1])  # Probability of class 1 (fraud) for 25 developments in PLA history

# Convert likelihoods lists to NumPy arrays
likelihoods_0 = np.array(likelihoods_0)
likelihoods_25 = np.array(likelihoods_25)

# Calculate the mean likelihoods for each group
mean_likelihood_0 = np.mean(likelihoods_0)
mean_likelihood_25 = np.mean(likelihoods_25)

print("Mean likelihood for 0 developments in PLA history:", mean_likelihood_0)
print("Mean likelihood for 25 developments in PLA history:", mean_likelihood_25)
#%%
# contacten_onderwerp_no_show // Contact subject client has not shown up for meeting
likelihoods_show = []
likelihoods_noshow = []

# Iterate through each sample in the test set
for index, row in X_test.iterrows():
    # Convert the row to a DataFrame to ensure it's a DataFrame object
    X_sample = pd.DataFrame(row).transpose()

    # Make predictions for a client that has shown up for meetings
    X_sample_show = X_sample.copy()
    X_sample_show['contacten_onderwerp_no_show'] = 0.0
    y_proba_show = pipeline.predict_proba(X_sample_show)

    # Make predictions for no show client
    X_sample_noshow = X_sample.copy()
    X_sample_noshow['contacten_onderwerp_no_show'] = 1.0
    y_proba_noshow = pipeline.predict_proba(X_sample_noshow)

    # Append the likelihoods for both age groups
    likelihoods_show.append(y_proba_show[:, 1])  # Probability of class 1 (fraud) for a client that has shown up for meetings
    likelihoods_noshow.append(y_proba_noshow[:, 1])  # Probability of class 1 (fraud) for no show client

# Convert likelihoods lists to NumPy arrays
likelihoods_show = np.array(likelihoods_show)
likelihoods_noshow = np.array(likelihoods_noshow)

# Calculate the mean likelihoods for each group
mean_likelihood_show = np.mean(likelihoods_show)
mean_likelihood_noshow = np.mean(likelihoods_noshow)

print("Mean likelihood for a client that has shown up for meetings:", mean_likelihood_show)
print("Mean likelihood for no show client:", mean_likelihood_noshow)
#%%
# instrument_ladder_huidig_activering // instrument ladder is currently activated
likelihoods_notactivated = []
likelihoods_activated = []

# Iterate through each sample in the test set
for index, row in X_test.iterrows():
    # Convert the row to a DataFrame to ensure it's a DataFrame object
    X_sample = pd.DataFrame(row).transpose()

    # Make predictions for a client without an activated instrument ladder
    X_sample_notactivated = X_sample.copy()
    X_sample_notactivated['instrument_ladder_huidig_activering'] = 0.0
    y_proba_notactivated = pipeline.predict_proba(X_sample_notactivated)

    # Make predictions for a client with an activated instrument ladder
    X_sample_activated = X_sample.copy()
    X_sample_activated['instrument_ladder_huidig_activering'] = 1.0
    y_proba_activated = pipeline.predict_proba(X_sample_activated)

    # Append the likelihoods for both age groups
    likelihoods_notactivated.append(y_proba_notactivated[:, 1])  # Probability of class 1 (fraud) for a client without an activated instrument ladder
    likelihoods_activated.append(y_proba_activated[:, 1])  # Probability of class 1 (fraud) for a client with an activated instrument ladder

# Convert likelihoods lists to NumPy arrays
likelihoods_notactivated = np.array(likelihoods_notactivated)
likelihoods_activated = np.array(likelihoods_activated)

# Calculate the mean likelihoods for each group
mean_likelihood_notactivated = np.mean(likelihoods_notactivated)
mean_likelihood_activated = np.mean(likelihoods_activated)

print("Mean likelihood for a client without an activated instrument ladder:", mean_likelihood_notactivated)
print("Mean likelihood for a client with an activated instrument ladder:", mean_likelihood_activated)
#%%
# instrument_reden_beeindiging_historie_succesvol // successful instrumentation history
likelihoods_not = []
likelihoods_successful = []

# Iterate through each sample in the test set
for index, row in X_test.iterrows():
    # Convert the row to a DataFrame to ensure it's a DataFrame object
    X_sample = pd.DataFrame(row).transpose()

    # Make predictions for a client without a successful instrumentation history
    X_sample_not = X_sample.copy()
    X_sample_not['instrument_reden_beeindiging_historie_succesvol'] = 0.0
    y_proba_not = pipeline.predict_proba(X_sample_not)

    # Make predictions for a client with a successful instrumentation history
    X_sample_successful = X_sample.copy()
    X_sample_successful['instrument_reden_beeindiging_historie_succesvol'] = 1.0
    y_proba_successful = pipeline.predict_proba(X_sample_successful)

    # Append the likelihoods for both age groups
    likelihoods_not.append(y_proba_not[:, 1])  # Probability of class 1 (fraud) for a client without a successful instrumentation history
    likelihoods_successful.append(y_proba_successful[:, 1])  # Probability of class 1 (fraud) for a client with a successful instrumentation history

# Convert likelihoods lists to NumPy arrays
likelihoods_not = np.array(likelihoods_not)
likelihoods_successful = np.array(likelihoods_successful)

# Calculate the mean likelihoods for each group
mean_likelihood_not = np.mean(likelihoods_not)
mean_likelihood_successful = np.mean(likelihoods_successful)

print("Mean likelihood for a client without a successful instrumentation history:", mean_likelihood_not)
print("Mean likelihood for a client with a successful instrumentation history:", mean_likelihood_successful)
#%%
# Input/output diversity:
# Create test set that spans the entire range of possible values for each feature and includes edge cases and outliers. How to do this? Look at most important features and span that range at least. Is the test set distribution of output in the same balance as the training set?

#%%
# Combinatorial/fairness testing: 
# Evaluate the model's performance and predictions separately for different demographic groups (e.g., age groups, genders) and compare the outcomes to detect any disparities or biases.
# assess whether the trained model treats certain groups differently.
# TODO create test case sets that test age groups, gender, if they have children, if they speak another language TODO add address   "adres", "woonadres", "verzendadres", "buurt", "wijk", "plaats", "persoon_geslacht_vrouw", persoon_leeftijd_bij_onderzoek, persoonlijke_eigenschappen_spreektaal_anders, relatie_kind_heeft_kinderen

age_groups = {
    'young_adult': [18, 30],  # 20-64 years old
    'youngish_adult': [31, 40],  # 20-64 years old
    'middle_aged_adult': [41, 50],  # 20-64 years old
    'older_adult': [51, 60],  # 20-64 years old
    'senior': [61, 120]  # 65+ years old (assuming 120 as upper limit)
}

results = {}
X_test_age = X_test.copy()

# Extract age information from the test set
X_test_age['age_group'] = pd.cut(X_test_age['persoon_leeftijd_bij_onderzoek'], bins=[0, 30, 40, 50, 60, 120], labels=['young_adult', 'youngish_adult', 'middle_aged_adult', 'older_adult', 'senior'])

# Evaluate model performance for each age group
for group in X_test_age['age_group'].unique():
    # Filter test set for the current age group
    X_group = X_test_age[X_test_age['age_group'] == group].drop(columns=['age_group'])
    y_group = y_test[X_test_age['age_group'] == group]

    # Predict using the model
    y_pred_group = pipeline.predict(X_group)

    # Calculate evaluation metrics
    accuracy_group = accuracy_score(y_group, y_pred_group)
    precision_group = precision_score(y_group, y_pred_group)
    recall_group = recall_score(y_group, y_pred_group)
    f1_score_group = f1_score(y_group, y_pred_group)

    # Store results for the current age group
    results[group] = {
        'Accuracy': accuracy_group,
        'Precision': precision_group,
        'Recall': recall_group,
        'F1 Score': f1_score_group
    }

# Print results for each age group
for group, metrics in results.items():
    print(f"Results for {group} age group:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print()

#%%
# Gender test

# Initialize a dictionary to store results for each gender group
results = {}
X_test_gender = X_test.copy()

# Extract gender information from the test set
X_test_gender['gender'] = X_test_gender['persoon_geslacht_vrouw'].apply(lambda x: 'woman' if x == 1.0 else 'man')

# Evaluate model performance for each gender group
for group in X_test_gender['gender'].unique():
    # Filter test set for the current gender group
    X_group = X_test_gender[X_test_gender['gender'] == group].drop(columns=['gender'])
    y_group = y_test[X_test_gender['gender'] == group]

    # Predict using the model
    y_pred_group = pipeline.predict(X_group)

    # Calculate evaluation metrics
    accuracy_group = accuracy_score(y_group, y_pred_group)
    precision_group = precision_score(y_group, y_pred_group)
    recall_group = recall_score(y_group, y_pred_group)
    f1_score_group = f1_score(y_group, y_pred_group)

    # Store results for the current gender group
    results[group] = {
        'Accuracy': accuracy_group,
        'Precision': precision_group,
        'Recall': recall_group,
        'F1 Score': f1_score_group
    }

# Print results for each gender group
for group, metrics in results.items():
    print(f"Results for {group} gender group:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print()
#%%
# Language test

# Initialize a dictionary to store results for each language group
results = {}
X_test_language = X_test.copy()

# Extract language information from the test set
X_test_language['language'] = X_test_language['persoonlijke_eigenschappen_spreektaal_anders'].apply(lambda x: 'other_language' if x == 1.0 else 'not')

# Evaluate model performance for each language group
for group in X_test_language['language'].unique():
    # Filter test set for the current language group
    X_group = X_test_language[X_test_language['language'] == group].drop(columns=['language'])
    y_group = y_test[X_test_language['language'] == group]

    # Predict using the model
    y_pred_group = pipeline.predict(X_group)

    # Calculate evaluation metrics
    accuracy_group = accuracy_score(y_group, y_pred_group)
    precision_group = precision_score(y_group, y_pred_group)
    recall_group = recall_score(y_group, y_pred_group)
    f1_score_group = f1_score(y_group, y_pred_group)

    # Store results for the current language group
    results[group] = {
        'Accuracy': accuracy_group,
        'Precision': precision_group,
        'Recall': recall_group,
        'F1 Score': f1_score_group
    }

# Print results for each language group
for group, metrics in results.items():
    print(f"Results for {group} language group:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print()
#%%
# children test

# Initialize a dictionary to store results for each children group
results = {}
X_test_children = X_test.copy()

# Extract gender information from the test set
X_test_children['children'] = X_test_children['relatie_kind_heeft_kinderen'].apply(lambda x: 'has_child' if x == 1.0 else 'not')

# Evaluate model performance for each children group
for group in X_test_children['children'].unique():
    # Filter test set for the current children group
    X_group = X_test_children[X_test_children['children'] == group].drop(columns=['children'])
    y_group = y_test[X_test_children['children'] == group]

    # Predict using the model
    y_pred_group = pipeline.predict(X_group)

    # Calculate evaluation metrics
    accuracy_group = accuracy_score(y_group, y_pred_group)
    precision_group = precision_score(y_group, y_pred_group)
    recall_group = recall_score(y_group, y_pred_group)
    f1_score_group = f1_score(y_group, y_pred_group)

    # Store results for the current children group
    results[group] = {
        'Accuracy': accuracy_group,
        'Precision': precision_group,
        'Recall': recall_group,
        'F1 Score': f1_score_group
    }

# Print results for each children group
for group, metrics in results.items():
    print(f"Results for {group} children group:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print()
#%%
# Age test
# Initialize variables to store counts
same_predictions_count = 0
total_samples = len(X_test)

# Iterate through each sample in the test set
for index, row in X_test.iterrows():
    # Convert the row to a DataFrame to ensure it's a DataFrame object
    X_sample = pd.DataFrame(row).transpose()

    # Make predictions for age 25
    X_sample_25 = X_sample.copy()
    X_sample_25['persoon_leeftijd_bij_onderzoek'] = 25
    y_pred_25 = pipeline.predict(X_sample_25)

    # Make predictions for age 65
    X_sample_65 = X_sample.copy()
    X_sample_65['persoon_leeftijd_bij_onderzoek'] = 65
    y_pred_65 = pipeline.predict(X_sample_65)

    # Check if predictions are the same
    if y_pred_25 == y_pred_65:
        same_predictions_count += 1

# Calculate the fraction of cases where the predictions are the same
fraction_same_predictions = same_predictions_count / total_samples

print("Fraction of cases where predictions are the same for age 25 and 65:", fraction_same_predictions)
#%%
# Gender test
# Initialize variables to store counts
same_predictions_count = 0
total_samples = len(X_test)

# Iterate through each sample in the test set
for index, row in X_test.iterrows():
    # Convert the row to a DataFrame to ensure it's a DataFrame object
    X_sample = pd.DataFrame(row).transpose()

    # Make predictions for men
    X_sample_men = X_sample.copy()
    X_sample_men['persoon_geslacht_vrouw'] = 0.0
    y_pred_men = pipeline.predict(X_sample_men)

    # Make predictions for women
    X_sample_women = X_sample.copy()
    X_sample_women['persoon_geslacht_vrouw'] = 1.0
    y_pred_women = pipeline.predict(X_sample_women)

    # Check if predictions are the same
    if y_pred_men == y_pred_women:
        same_predictions_count += 1

# Calculate the fraction of cases where the predictions are the same
fraction_same_predictions = same_predictions_count / total_samples

print("Fraction of cases where predictions are the same for men and women:", fraction_same_predictions)
#%%
# Language test
# Initialize variables to store counts
same_predictions_count = 0
total_samples = len(X_test)

# Iterate through each sample in the test set
for index, row in X_test.iterrows():
    # Convert the row to a DataFrame to ensure it's a DataFrame object
    X_sample = pd.DataFrame(row).transpose()

    # Make predictions for not
    X_sample_not = X_sample.copy()
    X_sample_not['persoonlijke_eigenschappen_spreektaal_anders'] = 0.0
    y_pred_not = pipeline.predict(X_sample_not)

    # Make predictions for other
    X_sample_other = X_sample.copy()
    X_sample_other['persoonlijke_eigenschappen_spreektaal_anders'] = 1.0
    y_pred_other = pipeline.predict(X_sample_other)

    # Check if predictions are the same
    if y_pred_not == y_pred_other:
        same_predictions_count += 1

# Calculate the fraction of cases where the predictions are the same
fraction_same_predictions = same_predictions_count / total_samples

print("Fraction of cases where predictions are the same for dutch speakers and non-dutch speakers:", fraction_same_predictions)
#%%
# Children test
# Initialize variables to store counts
same_predictions_count = 0
total_samples = len(X_test)

# Iterate through each sample in the test set
for index, row in X_test.iterrows():
    # Convert the row to a DataFrame to ensure it's a DataFrame object
    X_sample = pd.DataFrame(row).transpose()

    # Make predictions for not
    X_sample_not = X_sample.copy()
    X_sample_not['relatie_kind_heeft_kinderen'] = 0.0
    y_pred_not = pipeline.predict(X_sample_not)

    # Make predictions for other
    X_sample_children = X_sample.copy()
    X_sample_children['relatie_kind_heeft_kinderen'] = 1.0
    y_pred_children = pipeline.predict(X_sample_children)

    # Check if predictions are the same
    if y_pred_not == y_pred_children:
        same_predictions_count += 1

# Calculate the fraction of cases where the predictions are the same
fraction_same_predictions = same_predictions_count / total_samples

print("Fraction of cases where predictions are the same for people with or without children:", fraction_same_predictions)
#%%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score

# Assuming your pipeline steps and classifier are correctly defined
pipeline_steps = [
    ('scaler', StandardScaler()),  # First, scale the features
    ('pca', PCA(n_components=0.95)),  # Then, apply PCA to reduce the number of features
    ('classification', classifier)  # Finally, use the classifier for prediction
]

pipeline = Pipeline(steps=pipeline_steps)

# Define the parameter grid to search
param_grid = {
    'classification__n_estimators': [500, 600, 700],  # List of n_estimators to try
    'classification__learning_rate': [0.2, 0.3, 0.4, 0.5]  # List of learning rates to try
}

# Define the scoring metrics
scoring_metrics = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score), 'recall': make_scorer(recall_score), 'f1': make_scorer(f1_score)}

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring_metrics, refit='precision', cv=5, verbose=3, return_train_score=True)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy from grid search: {grid_search.best_score_}")

best_precision = grid_search.cv_results_['mean_test_precision'][grid_search.best_index_]
print(f"Precision for best accuracy: {best_precision}")