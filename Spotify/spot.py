import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

songs = pd.read_csv('Features.csv')
streams = pd.read_csv('Streams.csv')

df = songs.merge(streams, right_on='Song', left_on='name')

print(songs.head())
print(streams.head())
print(df.head())