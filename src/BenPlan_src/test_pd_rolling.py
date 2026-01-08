#!/Library/Frameworks/Python.framework/Versions/Current/bin/python3

"""
Test the pands DataFrame rolling function
"""
import numpy as np
import pandas as pd

WINDOW = 3
# Create a DataFrame 3 rows and 25 columns
df = pd.DataFrame(
    np.random.randint(low=0, high=11, size=(3, 25)), columns=list('ABCDEFGHIJKLMNOPQRSTUVWXY'))
print(df)
# Calculate the rolling sum of the DataFrame with a window size of 12
# Use .T because axis=1 is deprecated
df_rolling_sum = df.T.rolling(window=WINDOW, min_periods=WINDOW).sum()
print(df_rolling_sum)
