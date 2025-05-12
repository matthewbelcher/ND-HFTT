import pandas as pd 

effr_df = pd.read_csv('data/effr.csv', parse_dates=['Effective Date'])
effr_df.drop(columns=effr_df.columns[(effr_df.isna().sum() > 2000)].to_list(), inplace=True)
effr_df.drop(columns=['Rate Type'], inplace=True)
effr_df.rename(columns={'Effective Date': 'date', 'Rate (%)' : 'rate', 'Target Rate From (%)' : 'target_low', 'Target Rate To (%)' : 'target_high'}, inplace=True)
effr_df.to_csv('data/effr_cleaned.csv', index=False)