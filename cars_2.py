import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('cars.csv')
print(df.head())
print(df.columns)
corr = df.corrwith(df['price_usd']).sort_values()
print(corr)