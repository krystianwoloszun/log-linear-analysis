import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2
import numpy as np

titanic = sns.load_dataset("titanic")
df = titanic[["sex", "class", "survived"]].dropna()
df = df.rename(columns={"class": "passenger_class"})
# print(len(df))
# print(df.head())

counts = (
    df.groupby(["sex", "passenger_class", "survived"]).size().reset_index(name="count")
)
# print(counts)

# ~~~~~Independence Model~~~~~

model_ind = smf.glm(
    formula="count ~ sex + passenger_class + survived",
    data=counts,
    family=sm.families.Poisson(),
).fit()

print(model_ind.summary())

# ~~~~~Full Model~~~~~

model_inter = smf.glm(
    formula="count ~ sex * passenger_class * survived",
    data=counts,
    family=sm.families.Poisson(),
).fit()

print(model_inter.summary())

LR = model_ind.deviance - model_inter.deviance
df_diff = model_ind.df_resid - model_inter.df_resid

p_value = chi2.sf(LR, df_diff)

print("~~~~~Model comparison~~~~~")
print(f"LR = {LR:.4f}")
print(f"df = {df_diff}")
print(f"p = {p_value:.3e}")

# Convert log-scale coefficients into count multipliers
print(np.exp(model_inter.params))
