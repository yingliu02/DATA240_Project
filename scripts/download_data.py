is_churn_url = 'https://drive.google.com/file/d/1P43072y7KI4RUOKgWzZc3ov8kMaFQMVA/view?usp=sharing'
is_churn_url = 'https://drive.google.com/uc?id=' + is_churn_url.split('/')[-2]

# is_churn_url = 'is_churn.csv'
is_churn_df = pd.read_csv(is_churn_url)

# print(is_churn_df)
transactions_url = 'https://drive.google.com/file/d/1oPbL4uWPtFvq41IqNfMNW7QUaGA3INXV/view?usp=sharing'
transactions_url = 'https://drive.google.com/uc?id=' + transactions_url.split('/')[-2]
# transactions_url = 'transactions.csv'
transactions_df = pd.read_csv(transactions_url)

# print(transactions_df)
user_logs_url = 'https://drive.google.com/file/d/1FNMohIsfIcBFs2V-0m43qYoIGP1gCLQe/view?usp=sharing'
user_logs_url = 'https://drive.google.com/uc?id=' + user_logs_url.split('/')[-2]

user_logs_df = pd.read_csv(user_logs_url)

# print(user_logs_df)
members_url = 'https://drive.google.com/file/d/1yWY0mrpyEU3Ug98H8JsbQ-esROu5L5X6/view?usp=sharing'
members_url = 'https://drive.google.com/uc?id=' + members_url.split('/')[-2]

# members_path = 'members.csv'
members_df = pd.read_csv(members_url)

# print(members_df)
