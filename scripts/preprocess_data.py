merged_data[['gender']] = merged_data[['gender']].fillna('-1')

# Convert registration_init_time, transaction_date, membership_expire_date to years

merged_data['registration_init_time'] = merged_data['registration_init_time'].astype(str).str[:4]
merged_data['transaction_date'] = merged_data['transaction_date'].astype(str).str[:4]
merged_data['membership_expire_date'] = merged_data['membership_expire_date'].astype(str).str[:4]

print(merged_data)
# replacing values in column gender
merged_data['gender'].replace(['female', 'male'],
                        [0, 1], inplace=True)
# set row names (index) to the msno column

merged_data = merged_data.set_index('msno')

print(merged_data)
