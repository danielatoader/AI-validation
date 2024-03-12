#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import onnxruntime as rt
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from sklearn.preprocessing import StandardScaler
import pingouin as pg


# Temporarily adjust pandas display settings for large DataFrames
pd.set_option('display.max_rows', 100)  # Ensure 100 rows can be displayed
pd.set_option('display.max_columns', None)  # Ensure all columns can be displayed
pd.set_option('display.width', None)  # Automatically adjust display width to terminal size
pd.set_option('display.max_colwidth', None)  # Ensure full width of column content is shown
pd.set_option('display.float_format', '{:.4f}'.format)  # Format the float numbers for better readability

#%% md
# # Data preprocessing and feature selection
#%%
# Let's load the dataset
data = pd.read_csv('data/synth_data_for_training.csv')
#%%
# Calculate the correlation matrix
corr_matrix = data.corr()

# Initialize lists to store the results
highly_correlated_pairs = []

# Thresholds
threshold = 0.5
partial_threshold = 0.5

# Identify highly correlated pairs controlling for a third variable
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):  # i+1 to avoid self-correlation
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) > threshold:  # Check for high correlation
            for k in range(len(corr_matrix.columns)):
                if k != i and k != j:  # Exclude self-correlation and the current pair
                    # Calculate partial correlation
                    partial_corr = pg.partial_corr(data, x=corr_matrix.columns[i], y=corr_matrix.columns[j], covar=corr_matrix.columns[k])
                    if abs(partial_corr['r'].values[0]) < partial_threshold:
                        highly_correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value, corr_matrix.columns[k]))
                        break


# Determine features to remove
# Simple strategy: Remove one feature from each identified pair, preferentially keeping the one with lower average correlation with other features
features_to_remove = set()
for pair in highly_correlated_pairs:
    # Calculate average correlation of each feature with others
    avg_corr_i = corr_matrix[pair[0]].abs().mean()
    avg_corr_j = corr_matrix[pair[1]].abs().mean()

    # Prefer to remove the feature with higher average correlation
    if avg_corr_i > avg_corr_j:
        features_to_remove.add(pair[0])
    else:
        features_to_remove.add(pair[1])

# Create a new DataFrame excluding the features identified for removal
data_reduced = data.drop(columns=list(features_to_remove))

print(f"Original number of features: {data.shape[1]}, Reduced number of features: {data_reduced.shape[1]}")
#%%
# # # TODO: Jasper: Correlation does not imply causation. Two variables might be correlated due to a third variable or by coincidence.
# # # TODO: The threshold of 0.5 is arbitrary and might need to be adjusted based on your specific dataset and the model you are using. For some models, even moderately correlated features might pose problems, while for others, even higher correlations might not be as concerning.
# # # TODO: As a baseline model we can also use a model that has built-in mechanisms for feature selection (like L1 regularization for linear models). 
# # # TODO: Saga: Not checking missing values, outliers, or other data quality issues, imbalanced dataset. These can also affect the model's performance and should be addressed before or during feature selection.
# # 
# # Assuming data is your DataFrame
# # Calculate the correlation matrix
# corr_matrix = data.corr()
# 
# # Initialize lists to store the results
# highly_pos_correlated_pairs = []
# highly_neg_correlated_pairs = []
# 
# # Threshold for filtering high correlations (you can adjust this value)
# threshold = 0.5
# 
# # Iterate over the correlation matrix and store pairs of highly correlated features
# for i in range(len(corr_matrix.columns)):
#     for j in range(i+1, len(corr_matrix.columns)):  # i+1 to avoid self-correlation
#         if corr_matrix.iloc[i, j] > threshold:  # Positive correlation
#             highly_pos_correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
#         elif corr_matrix.iloc[i, j] < -threshold:  # Negative correlation
#             highly_neg_correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
# 
# # Sort the lists based on the correlation value
# highly_pos_correlated_pairs.sort(key=lambda x: x[2], reverse=True)
# highly_neg_correlated_pairs.sort(key=lambda x: x[2])
# 
# # Print out the highest positively and negatively correlated feature pairs
# print("Highly Positive Correlated Pairs:")
# for pair in highly_pos_correlated_pairs:
#     print(f"{pair[0]} and {pair[1]} with correlation {pair[2]:.2f}")
# 
# print("\nHighly Negative Correlated Pairs:")
# for pair in highly_neg_correlated_pairs:
#     print(f"{pair[0]} and {pair[1]} with correlation {pair[2]:.2f}")
# 
# # Assuming we choose to remove the second feature from each pair
# features_to_remove = {pair[1] for pair in highly_pos_correlated_pairs + highly_neg_correlated_pairs}
# 
# # Create a new DataFrame excluding the features identified for removal
# data_reduced = data.drop(columns=list(features_to_remove))
# 
# print(f"Original number of features: {data.shape[1]}, Reduced number of features: {data_reduced.shape[1]}")

#%% md
# test_size=0.20: This parameter specifies the proportion of the dataset to include in the test split. In this case, 20% of the data will be used for testing, and the remaining 80% will be used for training the model. The choice of test size affects model evaluation - too small a test set might not provide a representative evaluation of the model, while too large a test set might leave too little data for training, potentially leading to a poorly trained model.
# 
# random_state=42: This is a seed value for the random number generator. It ensures that the split between the training and testing sets is reproducible. Different seed values can result in different splits, which might lead to variations in model performance. Using a fixed random_state ensures that your results are reproducible, which is good for debugging and comparing models. However, relying solely on a single split can lead to overfitting to that specific partition of data, so it's often good practice to use cross-validation for more reliable estimates of model performance.
# 
# shuffle=True: This parameter indicates whether or not to shuffle the data before splitting. Shuffling is usually beneficial because it randomizes the distribution of data points across the training and testing sets, reducing the risk of biased splits. This is especially important if the data is ordered or clustered in some way that might influence learning if not randomized.
# 
# stratify=y: Stratifying means that the data is split in a way that preserves the same proportions of examples in each class as observed in the original dataset. This is crucial for imbalanced datasets, where one class significantly outnumbers the other(s). Without stratification, there's a risk that the training and testing sets might not accurately represent the class distribution, leading to skewed model evaluation and performance. Stratifying helps ensure that both training and test sets are representative of the overall dataset.
#%%
# Let's specify the features and the target
y = data_reduced['checked']
X = data_reduced.drop(['checked'], axis=1)
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

# Now, print the top 100 features with their importance
print(features_sorted.head(50))
#%%
# # List of features to keep:
# features_to_select = [
#     "pla_historie_ontwikkeling",
#     "ontheffing_dagen_hist_mean",
#     "belemmering_dagen_financiele_problemen",
#     "afspraak_aantal_woorden",
#     "deelname_act_reintegratieladder_werk_re_integratie",
#     "adres_aantal_brp_adres",
#     "contacten_onderwerp__arbeids_motivatie"
# ]
# 
# # Manually select the features from your dataframe
# X_selected = data_reduced[features_to_select]

# take first 10 features
X_selected = X.iloc[:, :10]
#%% md
# # Feature scaling and model training
#%%
# Define a gradient boosting classifier
classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
#%%
# Create a pipeline object with our selector and classifier
# NOTE: You can create custom pipeline objects but they must be registered to onnx or it will not recognise them
# Because of this we recommend using the onnx known objects as defined in the documentation
# TODO: The pipeline construction and inclusion of feature scaling via StandardScaler is a good practice, ensuring that your model is not biased by the scale of the features.
pipeline_steps = [
    ('scaling', StandardScaler()),
    ('classification', RandomForestClassifier())
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
