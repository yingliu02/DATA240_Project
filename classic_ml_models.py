import pandas as pd
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

CLASSIFIER_NAMES = ["decision_tree", "adaboost", "random_forest", "xgboost", "knn"]

class MLClassifier:
    def __init__(self, classifier_name, x_train, y_train, x_test, y_test):
        self.classifier_name = classifier_name
        if classifier_name == "decision_tree":
             self.classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
        elif classifier_name == "adaboost":
             self.classifier = AdaBoostClassifier(n_estimators=100, random_state=42)
        elif classifier_name == "random_forest":
             self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier_name == "xgboost":
             self.classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)
        elif classifier_name == "knn":
             self.classifier = KNeighborsClassifier()
        else:
            self.classifier = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def fit_and_get_metrics(self):
        # Fit the model
        model = self.classifier.fit(self.x_train, self.y_train)

        # Predict on the test set
        y_pred = model.predict(self.x_test)

        # Get the confusion matrix
        self.cm = confusion_matrix(self.y_test, y_pred)

        # Get the accuracy score
        self.acc = accuracy_score(self.y_test, y_pred)

        # Get the precision score
        self.prec = precision_score(self.y_test, y_pred)

        # Get the recall score
        self.rec = recall_score(self.y_test, y_pred)

        # Get the F1 score
        self.f1 = f1_score(self.y_test, y_pred)

        # Get the ROC-AUC score
        self.roc_auc = roc_auc_score(self.y_test, y_pred)

        # Get the ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred)

        print(f'Accuracy for {self.classifier_name} Classifier: {self.acc:.4f}')
        print(f'Precision for {self.classifier_name} Classifier: {self.prec:.4f}')
        print(f'Recall for {self.classifier_name} Classifier: {self.rec:.4f}')
        print(f'F1 Score for {self.classifier_name} Classifier: {self.f1:.4f}')
        print(f'ROC-AUC Score for {self.classifier_name} Classifier: {self.roc_auc:.4f}')
        print()

    
    def plot_model_results(self):
        y_pred = self.classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(12, 5))

        # Plotting Confusion Matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion Matrix for {model_name}'.format(model_name=self.classifier_name))
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # # Plotting Accuracy Score
        # plt.subplot(1, 2, 2)
        # plt.bar(model_name, acc)
        # plt.title(f'Accuracy Score for {model_name}')
        # plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self):
        assert self.classifier_name in ("ada_boost", "random_forest", "xgboost")
        # Plot feature importance using mean decrease in impurity
        importances = self.classifier.feature_importances_

        feature_importances = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
        print(feature_importances)

        fig, ax = plt.subplots()
        feature_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances for {model_name} using MDI".format(model_name=self.classifier_name))
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()


def read_merged_data():
    merged_raw_data_url = 'https://drive.google.com/file/d/1WDfh8HLYOtUNuhRZqKCScd1qb4l9sqyj/view?usp=sharing'
    merged_raw_data_url = 'https://drive.google.com/uc?id=' + merged_raw_data_url.split('/')[-2]

    churn_df = pd.read_csv(merged_raw_data_url)

    return churn_df


def split_data():

    # Identify features and target variable
    X = churn_df.drop(['is_churn', 'msno'], axis=1)
    y = churn_df['is_churn']

    yes_count = churn_df['is_churn'].value_counts().get(1, 0)
    no_count = churn_df['is_churn'].value_counts().get(0, 0)
    print(f"count of churned users: {yes_count}")
    print(f"count of non-churned users: {no_count}")

    yes_percent = (yes_count / (yes_count + no_count)) * 100
    no_percent = (no_count / (yes_count + no_count)) * 100

    print(f"Percentage of churned users: {yes_percent:.2f}%")
    print(f"Percentage of non-churned users: {no_percent:.2f}%")

    # Split the dataset to training, validation and test dataset with ratio 6:2:2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

    # Combine X_train and y_train into a single DataFrame for undersampling
    train_data = pd.concat([X_train, y_train], axis=1)

    # Identify the minority class label
    minority_class_label = train_data['is_churn'].value_counts().idxmin()

    # Apply random undersampling on imbalanced target data
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_balanced, y_balanced = undersampler.fit_resample(train_data.drop('is_churn', axis=1), train_data['is_churn'])
    print(y_balanced.value_counts())

    return X_train, y_train, X_balanced, y_balanced, X_test, y_test


if __name__ == '__main__':
    churn_df = read_merged_data()

    print(churn_df)

    X_train, y_train, X_balanced, y_balanced, X_test, y_test = split_data()

    # Imbalanced
    classifiers_for_imbalanced_data = {}
    for classifier_name in CLASSIFIER_NAMES:
        print(f'Evaluation Matrix for Imbalanced Data:')
        classifiers_for_imbalanced_data[classifier_name] = MLClassifier(classifier_name, X_train, y_train, X_test, y_test)
        classifiers_for_imbalanced_data[classifier_name].fit_and_get_metrics()
        classifiers_for_imbalanced_data[classifier_name].plot_model_results()
        if classifier_name in ("ada_boost", "random_forest", "xgboost"):
            classifiers_for_imbalanced_data[classifier_name].plot_feature_importance()

    # Balanced
    classifiers_for_balanced_data = {}
    for classifier_name in CLASSIFIER_NAMES:
        print(f'Evaluation Matrix for Balanced Data:')
        classifiers_for_balanced_data[classifier_name] = MLClassifier(classifier_name, X_balanced, y_balanced, X_test, y_test)
        classifiers_for_balanced_data[classifier_name].fit_and_get_metrics()
        classifiers_for_balanced_data[classifier_name].plot_model_results()
        if classifier_name in ("ada_boost", "random_forest", "xgboost"):
            classifiers_for_balanced_data[classifier_name].plot_feature_importance()

    




