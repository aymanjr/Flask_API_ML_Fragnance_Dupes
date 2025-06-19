import pandas as pd

# Force known safe encoding for French-style accented characters
ENCODING = "ISO-8859-1"  # or try "cp1252" if this fails

df = pd.read_csv("fra_cleaned.csv", sep=";", encoding=ENCODING)
print(df.head())
