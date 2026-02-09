import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("data\FINAL_PROJECT_DATASET.csv")

"""EDA"""

print(df["Strength"].describe())

#Outlier Detection
sns.boxplot(x=df["Strength"])
plt.title("Outlier detection for Compression strength occupied in N days")
plt.show()

#Correlation matrixes
attributes = ["Cement","GGBS", "FlyAsh", "Water", "CoarseAggregate", "Sand", "Admixture","age"]
strength_cols = ["Strength"]
for i in attributes:
    print(f"Min of {i}: {df[i].min()}, Max of {i}: {df[i].max()}")


corr = df[attributes + strength_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Materials and Strength")
plt.show()
    

"""Feature Engineering"""
#Creating features DataFrame
df2=pd.DataFrame()

# Total binder (cementitious materials)
df2["Binder"] = df["Cement"] + df["GGBS"] + df["FlyAsh"]

# Water–binder ratio
df2["WBRatio"] = df["Water"] / df2["Binder"]

# SCM contribution ratios
df2["FA_ratio"] = df["FlyAsh"] / df2["Binder"]
df2["GGBS_ratio"] = df["GGBS"] / df2["Binder"]

# Aggregate packing structure
df2["Sand_ratio"] = df["Sand"] / (df["Sand"] + df["CoarseAggregate"])

# Aggregate skeleton strength
df2["Agg_Binder"] = (df["Sand"] + df["CoarseAggregate"]) / df2["Binder"]

# Paste volume (paste vs aggregate balance)
df2["Paste_volume"] = (df["Water"] + df2["Binder"]) / (
    df["Water"] + df2["Binder"] + df["Sand"] + df["CoarseAggregate"]
)
df2["Strength"] = df["Strength"]
df2["age"]=df["age"]
# Save df_copy to a CSV file
df2.to_csv("data/features.csv", index=False)