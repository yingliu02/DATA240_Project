import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


def read_merged_data():
    merged_raw_data_url = 'https://drive.google.com/file/d/1WDfh8HLYOtUNuhRZqKCScd1qb4l9sqyj/view?usp=sharing'
    merged_raw_data_url = 'https://drive.google.com/uc?id=' + merged_raw_data_url.split('/')[-2]

    churn_df = pd.read_csv(merged_raw_data_url)

    return churn_df

def display_merged_data(churn_df):

    churn_df = churn_df.drop(['msno'], axis=1)

    print(churn_df)


    # Correlation Analysis
    correlations = churn_df.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation with Target Variable')
    plt.show()

    # Assuming registration_init_time is in the format YYYY
    churn_df['registration_year'] = pd.to_datetime(churn_df['registration_init_time'], format='%Y')

    # Group by registration year and calculate churn rate
    churn_over_years = churn_df.groupby(churn_df['registration_year'].dt.year)['is_churn'].mean()

    churn_over_years.plot(kind='bar')
    plt.title('Churn Rate by Registration Year')
    plt.xlabel('Year')
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.show()

    #churn by city
    city_churn = churn_df.groupby('city')['is_churn'].mean().sort_values()
    city_churn.plot(kind='bar', figsize=(10, 5))
    plt.title('Churn by City')
    plt.xlabel('City')
    plt.ylabel('Churn Rate')
    plt.show()

    # Replace 'gender_integers' with the actual integers representing genders in your dataset
    sns.countplot(x='gender', hue='is_churn', data=churn_df)
    plt.title('Churn by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    #plt.xticks(range(len(gender)), ['Gender1', 'Gender2'])  # Replace with actual gender names
    plt.show()

    #churn by payment method
    payment_method_churn = churn_df.groupby('payment_method_id')['is_churn'].mean().sort_values()
    payment_method_churn.plot(kind='bar', figsize=(10, 5))
    plt.title('Churn by Payment Method')
    plt.xlabel('Payment Method ID')
    plt.ylabel('Churn Rate')
    plt.show()

    #churn by auto renewal
    sns.countplot(x='is_auto_renew', hue='is_churn', data=churn_df)
    plt.title('Churn by Auto-Renew Status')
    plt.xlabel('Auto-Renew Status')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.show()

if __name__ == '__main__':
    churn_df = read_merged_data()

    print(churn_df)

    display_merged_data(churn_df)

    




