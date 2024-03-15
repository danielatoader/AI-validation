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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from skl2onnx import convert_sklearn
from sklearn.preprocessing import StandardScaler
import pingouin as pg
import pandas as pd
from scipy import stats
import numpy as np
# define a XGBoost classifier
import xgboost as xgb
import warnings

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
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data/synth_data_for_training.csv')

print("Before cleaning:")
print("Missing values per column:")
print("Total missing values:", data.isna().sum().sum())

# Identify non-binary columns
non_binary_columns = [col for col in data.columns if not (np.isin(data[col].unique(), [0, 1]).all() and len(data[col].unique()) == 2)]

# Calculate Z-scores for non-binary columns only
z_scores_non_binary = np.abs(stats.zscore(data[non_binary_columns], nan_policy='omit'))

# Mask to identify rows with outliers in non-binary columns
outlier_mask = (z_scores_non_binary > 3).any(axis=1)

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
def calculate_correlation_matrix(data):
    return data.corr()

def identify_strong_correlations(corr_matrix, threshold=0.3):
    strong_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                strong_pairs.append((i, j, corr_value))
    return strong_pairs

def evaluate_partial_correlation(data, corr_matrix, strong_pairs, partial_threshold=0.2):
    highly_correlated_pairs = []
    for i, j, corr_value in strong_pairs:
        for k in range(len(corr_matrix.columns)):
            if k != i and k != j:
                subset = data[[corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.columns[k]]].dropna()
                if subset.shape[0] >= 3:
                    partial_corr = pg.partial_corr(data, x=corr_matrix.columns[i], y=corr_matrix.columns[j], covar=corr_matrix.columns[k])
                    if abs(partial_corr['r'].values[0]) < partial_threshold:
                        highly_correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value, corr_matrix.columns[k]))
                        break
                else:
                    print(f"Skipping due to insufficient data: {corr_matrix.columns[i]}, {corr_matrix.columns[j]}, {corr_matrix.columns[k]}")
    return highly_correlated_pairs

def find_highly_correlated_pairs(data, corr_matrix, threshold=0.5, partial_threshold=0.5):
    strong_pairs = identify_strong_correlations(corr_matrix, threshold)
    highly_correlated_pairs = evaluate_partial_correlation(data, corr_matrix, strong_pairs, partial_threshold)
    return highly_correlated_pairs

def determine_features_to_remove(corr_matrix, highly_correlated_pairs):
    features_to_remove = set()
    for i, j, _, _ in highly_correlated_pairs:
        avg_corr_i = corr_matrix[i].abs().mean()
        avg_corr_j = corr_matrix[j].abs().mean()
        if avg_corr_i > avg_corr_j:
            features_to_remove.add(i)
        else:
            features_to_remove.add(j)
    return features_to_remove

def reduce_features_from_data(data, features_to_remove):
    return data.drop(columns=list(features_to_remove))

def main(data):
    corr_matrix = calculate_correlation_matrix(data)
    highly_correlated_pairs = find_highly_correlated_pairs(data, corr_matrix)
    features_to_remove = determine_features_to_remove(corr_matrix, highly_correlated_pairs)
    data_reduced = reduce_features_from_data(data, features_to_remove)
    print(f"Original number of features: {data.shape[1]}, Reduced number of features: {data_reduced.shape[1]}")
    return data_reduced


data_reduced = main(data)
#%%
# Check how imbalance the dataset is
data_reduced['checked'].value_counts(normalize=True)
#%%
# Let's specify the features and the target
y = data_reduced['checked']
X = data_reduced.drop(['checked'], axis=1)
X = X.astype(np.float32)

# TODO: Instead of a single train-test split, consider using cross-validation to assess model performance more robustly. This approach can help ensure the model's generalizability across different subsets of our data.
# Let's split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
#%%
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Check how imbalance the dataset is after SMOTE
y_train.value_counts(normalize=True)
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

X_selected = X.iloc[:, :20]
#%% md
# # Feature scaling and model training
#%%
classifier = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=(len(y)-sum(y))/sum(y),  # Adjusting for imbalance
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
    ('scaling', StandardScaler()),
    ('classification', classifier)
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
