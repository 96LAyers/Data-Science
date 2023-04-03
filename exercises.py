import pandas as pd

df = pd.DataFrame([["A", 1], ["B", 2], ["C", 3], ["D", 4]],
		  columns = ["Col_A", "Col_B"])

sort_list = ["C", "D", "B", "A"]

print(df)

df = pd.DataFrame([["A", 1], ["B", 2], ["C", 3], ["D", 4]],
                  columns = ["Col_A", "Col_B"])

new_column = ["P", "Q", "R", "S"]
insert_position = 1

df['Col_C'] = new_column
df = df[['Col_A','Col_C','Col_B']]

print(df)
"""
	col_A	col_C	col_B
0	A	P	1
1	B	Q	2
2	C	R	3
3	D	S	4
"""

df = pd.DataFrame([["A", 1, True], ["B", 2, False],
                   ["C", 3, False], ["D", 4, True]], 
                  columns=["col_A", "col_B", "col_C"])

dt_type = "bool"

df_output = df.select_dtypes(include=dt_type)
print(df_output)

"""
	col_C
0	True
1	False
2	False
3	True
"""
#report number of non-NaN values in column
import numpy as np

df = pd.DataFrame([["A", np.NaN], [np.NaN, 2],
                   ["C", np.NaN], ["D", 4]], 
                  columns=["col_A", "col_B"])

"""
	col_A	col_B
0	A	NaN
1	NaN	2.0
2	C	NaN
3	D	4.0
"""

output_df = df.count()
print(output_df)
"""
col_A    3
col_B    2
"""
#Split data into equal parts
df = pd.DataFrame([["A", 1], ["B", 2], ["C", 3], ["D", 4]], 
                  columns=["col_A", "col_B"])

parts = 2
count = len(df)

out_df1, out_df2 = np.split(df, parts)
print(out_df1)
print(out_df2)

print(df)
print(df.loc[:,'col_A'::])
print(df.iloc[:,1::])
print(df.iloc[:,::2])