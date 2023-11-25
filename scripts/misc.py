# bar plot for number of churn

count_churn = len(is_churn_df[(is_churn_df['is_churn'] == 1)])
count_renewal = len(is_churn_df[(is_churn_df['is_churn'] == 0)])

x = ['churn', 'renewal']
y = [count_churn, count_renewal]

plt.bar(x, y)

plt.xlabel("Churn or not")
plt.ylabel("Count of members")
plt.title("Number of Churn vs. renewal members")
plt.show()
