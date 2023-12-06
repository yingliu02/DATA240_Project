import argparse
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import optuna
import pytorch_lightning as pl
import os
import pandas as pd

def resample_data(X, y):
    # Resample the data
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled

def split_data(X, y):
    # Split the data into training and testing sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

def pull_data():
    if not os.path.exists('data'):
        os.makedirs('data')
    # check if data is already downloaded
    if os.path.exists('data/merged_raw_data.csv'):
        df = pd.read_csv('data/merged_raw_data.csv')
    else:
        merged_raw_data_url = 'https://drive.google.com/file/d/1WDfh8HLYOtUNuhRZqKCScd1qb4l9sqyj/view?usp=sharing'
        merged_raw_data_url = 'https://drive.google.com/uc?id=' + merged_raw_data_url.split('/')[-2]
        df = pd.read_csv(merged_raw_data_url)
        df.drop('msno', axis=1, inplace=True)
        df.to_csv('data/merged_raw_data.csv')
    X = df.drop(['is_churn'], axis=1)
    y = df['is_churn']

    return X, y



if __name__ == '__main__':
    model_name = input("Enter model name (simple_nn or enhanced_nn): ")
    balanced = input("Do you want the data to be balanced? (True or False): ")

    balanced = balanced.lower() in ['true', '1', 'yes']

    X, y = pull_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    if balanced:
        X_train, y_train = resample_data(X_train, y_train)
    
    if model_name == 'simple_nn':
        from simple_nn import train_dataloader, test_dataloader, ChurnPredictor
        # Train the model
        model = ChurnPredictor(input_size=14, hidden_size=20, output_size=1, batch_size=64)
        trainer = pl.Trainer(max_epochs=20)
        [train_data, val_data] = train_dataloader(X_train, y_train, X_val, y_val)
        trainer.fit(model, train_data, val_data)
        # Test the model
        trainer.test(model, dataloaders=test_dataloader(X_test, y_test))
    elif model_name == 'enhanced_nn':
        from enhanced_nn import objective
        # Run a hyperparameter optimization study to minimize the validation loss
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial), n_trials=100)
        print(resampled_study.best_trial)
    else:
        print('invalid model name, must be simple_nn or enhanced_nn')
        exit()
    
    
