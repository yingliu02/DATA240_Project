merged_raw_data_url = 'https://drive.google.com/file/d/1WDfh8HLYOtUNuhRZqKCScd1qb4l9sqyj/view?usp=sharing'
merged_raw_data_url = 'https://drive.google.com/uc?id=' + merged_raw_data_url.split('/')[-2]

df = pd.read_csv(merged_raw_data_url)

# save the data
df.to_csv('data/merged_raw_data.csv', index=False)
