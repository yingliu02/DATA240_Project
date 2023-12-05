import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

TRAIN_URL_BASE = 'https://drive.google.com/file/d/1NTnEDxlG9KBnr9rUXp5MdXSvrMpWY4fW/view?usp=sharing'

def get_train_data_df():
    train_url = 'https://drive.google.com/uc?id=' + TRAIN_URL_BASE.split('/')[-2]

    train_df = pd.read_csv(train_url)
    transactions_df = pd.read_csv('transactions.csv')
    members_df = pd.read_csv('members_v3.csv')
    user_logs_df = pd.read_csv('user_logs.csv',nrows=700000)

    return train_df, members_df, transactions_df, user_logs_df

def get_merged_data():

    # count the average playtime per day for every member
    avg_playtime = user_logs_df.groupby('msno', as_index=False)['total_secs'].mean()
    avg_playtime = avg_playtime.rename(columns={"total_secs": "playtime_per_day"})
    print(avg_playtime.min())

    # get the information about the latest transaction
    latest_transaction_date = transactions_df.groupby('msno', as_index=False)['transaction_date'].max()
    latest_transactions = transactions_df.merge(latest_transaction_date, on=["msno", "transaction_date"])
    # print(latest_transactions)

    # merge all dataset
    merged_data = train_df.merge(avg_playtime, how='inner', on=["msno"]).merge(members_df, how='inner', on="msno").merge(latest_transactions, how='inner', on=["msno"])

    print(merged_data)
    return merged_data

def process_and_save_merged_data():
    
    # check missing values
    merged_data.isna().sum(axis = 0)


    merged_data[['gender']] = merged_data[['gender']].fillna('-1')

    # Convert registration_init_time, transaction_date, membership_expire_date to years
    merged_data['registration_init_time'] = merged_data['registration_init_time'].astype(str).str[:4]
    merged_data['transaction_date'] = merged_data['transaction_date'].astype(str).str[:4]
    merged_data['membership_expire_date'] = merged_data['membership_expire_date'].astype(str).str[:4]

    # replacing values in column gender
    merged_data['gender'].replace(['female', 'male'],
                            [0, 1], inplace=True)
    
    # convert all columns datatype to numeric
    merged_data['gender'] = merged_data['gender'].astype(int)
    merged_data['registration_init_time'] = merged_data['registration_init_time'].astype(int)
    merged_data['transaction_date'] = merged_data['transaction_date'].astype(int)
    merged_data['membership_expire_date'] = merged_data['membership_expire_date'].astype(int)

    # set row names (index) to the msno column

    merged_data = merged_data.set_index('msno')

    merged_data.to_csv('merged_raw_datasets.csv', encoding='utf-8')

    return merged_data

def plot_data_distribution():
    # bar plot for number of churn
    count_churn = len(merged_data[(merged_data['is_churn'] == 1)])
    count_renewal = len(merged_data[(merged_data['is_churn'] == 0)])

    x = ['churn', 'renewal']
    y = [count_churn, count_renewal]

    print(f'{x[0]}: {y[0]}')
    print(f'{x[1]}: {y[1]}')

    plt.bar(x, y)

    plt.xlabel("Churn or not")
    plt.ylabel("Count of members")
    plt.title("Number of Churn vs. renewal members")
    plt.show()


if __name__ == '__main__':

    train_df, members_df, transactions_df, user_logs_df = get_train_data_df()

    # merge all dataset
    merged_data = get_merged_data()

    # process merged data
    processed_merged_data = process_and_save_merged_data()

    plot_data_distribution()
