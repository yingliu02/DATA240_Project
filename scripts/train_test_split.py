# Identify features and target variable
X = merged_data.drop(['is_churn'], axis=1)
y = merged_data['is_churn']

yes_count = merged_data['is_churn'].value_counts().get(1, 0)
no_count = merged_data['is_churn'].value_counts().get(0, 0)
print(yes_count)
print(no_count)

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Combine X_train and y_train into a single DataFrame for undersampling
train_data = pd.concat([X_train, y_train], axis=1)

# Identify the minority class label
minority_class_label = train_data['is_churn'].value_counts().idxmin()

# Apply random undersampling on imbalanced target data
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(train_data.drop('is_churn', axis=1), train_data['is_churn'])
print(X_resampled)
