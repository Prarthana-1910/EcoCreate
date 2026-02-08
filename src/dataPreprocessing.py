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
df2=df.copy()

#WBRatio
df2["WBRatio"]=df["Water"]/(df["Cement"]+df["FlyAsh"]+df["GGBS"])

# Save df_copy to a CSV file
df2.to_csv("data/features.csv", index=False)