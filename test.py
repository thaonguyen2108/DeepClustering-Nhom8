from pipeline import load_tabular_dataset

inget = load_tabular_dataset("./DataSet/shopping_behavior_updated.csv")

df = inget.dataframe

df.head()
df.info()