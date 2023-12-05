# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Manipulation
import pandas as pd
import numpy as np

# Date and Time
from datetime import datetime

# Additional visualization (not direct equivalents, but useful)
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates


if __name__ == '__main__':

    members = pd.read_csv("/Users/liuying/Downloads/members_v3.csv")

    print(members.head())
    members.describe()
    members.info()
    members.isnull().sum()


    # Set unrealistic ages to None
    min_age, max_age = 0, 100
    members.loc[(members['bd'] < min_age) | (members['bd'] > max_age), 'bd'] = None

    print(members['bd'])
    members.describe()

    # Convert date columns to datetime
    members['registration_init_time'] = pd.to_datetime(members['registration_init_time'], format='%Y%m%d')
    members.head()


    members.describe(include=[np.number, 'category', 'datetime'])

    # Check for Null Values
    null_count = members['registration_init_time'].isnull().sum()
    print(f"Number of null values in 'registration_init_time': {null_count}")

    # Inspect Unique Values
    unique_values = members['registration_init_time'].unique()
    print(f"Unique values sample in 'registration_init_time': {unique_values[:10]}")

    # Range Check
    date_min = members['registration_init_time'].min()
    date_max = members['registration_init_time'].max()
    print(f"Date range in 'registration_init_time': {date_min} to {date_max}")

    # Type Check
    all_dates = all(isinstance(x, pd.Timestamp) for x in members['registration_init_time'])
    print(f"All entries are datetime objects: {all_dates}")

    # Analyze the distribution of values
    print("City value counts:\n", members['city'].value_counts())
    print("\nRegistered_via value counts:\n", members['registered_via'].value_counts())


    # Apply one-hot encoding
    members_encoded = pd.get_dummies(members, columns=['city', 'registered_via'])

    # Display the first few rows of the modified DataFrame
    print(members_encoded.head())

    print(members["bd"].describe())
    members.bd.value_counts()

    print(f'{(members["gender"].isna().sum() / len(members["gender"])):.2%} of users did not specify their gender')

    """should we apply one hot encoding , in the github they have used it ,
    or should we simply drop or replace it with -1 , this is what we used initially
    for now keeping as is"""

    members.head()

    transac = pd.read_csv('/Users/liuying/Downloads/transactions.csv')
    transac.head()
    transac.info()

    #changing the transaction_date into date time format as we did in the previous members.csv file
    transac["transaction_date"] = pd.to_datetime(transac.transaction_date, format='%Y%m%d')
    transac.head()


    # Descriptive statistics
    print(transac.describe())

    # Histograms for numeric columns
    transac.hist(bins=50, figsize=(20,15))
    plt.show()

    # Convert 'transaction_date' and 'membership_expire_date' to datetime
    transac['transaction_date'] = pd.to_datetime(transac['transaction_date'], format='%Y%m%d')
    transac['membership_expire_date'] = pd.to_datetime(transac['membership_expire_date'], format='%Y%m%d')

    # Descriptive statistics for numeric columns
    print("Descriptive Statistics:\n", transac.describe())

    # Analyzing the distribution of categorical columns
    print("\nPayment Method Distribution:\n", transac['payment_method_id'].value_counts())
    print("\nAuto Renew Distribution:\n", transac['is_auto_renew'].value_counts())
    print("\nCancel Distribution:\n", transac['is_cancel'].value_counts())



    # Saving the modified DataFrame, if needed
    # transactions_df.to_csv('path_to/modified_transactions.csv', index=False)

    # Exploring relationships - for example, comparing 'plan_list_price' and 'actual_amount_paid'
    pd.plotting.scatter_matrix(transac[['plan_list_price', 'actual_amount_paid']], alpha=0.2, figsize=(6, 6))


    # Exploring trends over time
    transac.groupby(transac['transaction_date'].dt.to_period('M')).size().plot(kind='line', title='Transactions Over Time')


    # Assuming you have a DataFrame named 'trans' with a 'transaction_date' column

    # Filter the data for transactions after January 1, 2015
    trans_filtered = transac[transac['transaction_date'] > '2015-01-01']

    # Create a frequency plot for 'transaction_date'
    plt.figure(figsize=(10, 5))
    plt.hist(trans_filtered['transaction_date'], bins=pd.date_range(start='2015-01-01', end='2023-01-01', freq='1D'), color='red', alpha=0.5, label='Transaction Date')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xlabel('Transaction Date')
    plt.ylabel('Frequency')
    plt.title('Transaction Date Distribution')
    plt.legend(loc='upper right')
    # Set the x-axis to show every year and month
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc='upper right')
    plt.show()

    unique_transactions_user_count = transac["msno"].nunique()
    print(f"Unique users in transaction database: {unique_transactions_user_count:,}")

    logs = pd.read_csv('/Users/liuying/Downloads/user_logs.csv',nrows=700000)
    logs.info()
    logs.describe()

    logs['date'] = pd.to_datetime(logs['date'], format='%Y%m%d')
    logs[['num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs']].describe()
    

    # Plot 1: Entries per user
    p1_data = logs['msno'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.countplot(x=p1_data, color='blue')
    plt.xlabel('Entries per user')
    plt.title('Distribution of Log Entries per User')
    plt.show()


    logs_filtered = logs[abs(logs['total_secs']) < 1e5]
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=logs_filtered, x='total_secs', fill=True, bw_adjust=0.5, log_scale=True)
    plt.axvline(x=logs['total_secs'].median(), color='k', linestyle='--')
    plt.title('Density of Total Seconds (Limited to 100k Seconds)')
    plt.show()


    plt.figure(figsize=(10, 6))
    sns.histplot(data=logs, x='num_unq', binwidth=0.05, color='red', log_scale=True)
    plt.axvline(x=logs['num_unq'].median(), color='k', linestyle='--')
    plt.title('Distribution of Unique Number of Songs Played')
    plt.show()

    melted_logs = pd.melt(logs, id_vars=['msno'], value_vars=['num_25', 'num_50', 'num_75', 'num_985', 'num_100'], var_name='slen', value_name='cases')
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=melted_logs, x='cases', hue='slen', common_norm=False, fill=True, bw_adjust=0.1)
    plt.xscale('log')
    plt.xlim(1, 800)
    plt.xlabel('Number of Songs')
    plt.title('Density of Songs Played by % Played')
    plt.show()

    churn = pd.read_csv("/Users/liuying/Downloads/train.csv")
    #churn.rename({"msno": "user_id"}, axis=1, inplace=True)
    #churn["is_churn"] = churn.is_churn.astype(np.uint8)
    churn.head(10)

    churn.tail(10)

    churn_distribution = churn['is_churn'].value_counts()
    sns.barplot(x=churn_distribution.index, y=churn_distribution.values)
    plt.title('Distribution of Churn')
    plt.xlabel('Churn Status')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Not Churned', 'Churned'])
    plt.show()














