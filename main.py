import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt

titanic = sns.load_dataset("titanic")

df = titanic[["sex", "class", "survived"]].dropna()
df = df.rename(columns={"class": "passenger_class"})

counts = (
    df.groupby(["sex", "passenger_class", "survived"]).size().reset_index(name="count")
)

# ~~~~~ Independence Model ~~~~~

model_ind = smf.glm(
    formula="count ~ sex + passenger_class + survived",
    data=counts,
    family=sm.families.Poisson(),
).fit()

print(model_ind.summary())

# ~~~~~ Full Model ~~~~~

model_inter = smf.glm(
    formula="count ~ sex * passenger_class * survived",
    data=counts,
    family=sm.families.Poisson(),
).fit()

print(model_inter.summary())

# ~~~~~ Model comparison ~~~~~

LR = model_ind.deviance - model_inter.deviance
df_diff = model_ind.df_resid - model_inter.df_resid

p_value = chi2.sf(LR, df_diff)

print("~~~~~ Model comparison ~~~~~")
print(f"LR = {LR:.4f}")
print(f"df = {df_diff}")
print(f"p = {p_value:.3e}")

# Convert log-scale coefficients into count multipliers
print(np.exp(model_inter.params))


# ~~~~~ VISUALIZATIONS ~~~~~

# ~~~~~ 1. Survival by class + sex ~~~~~

plot_df = counts.copy()

plot_df["group"] = plot_df["passenger_class"].astype(str) + " | " + plot_df["sex"]

pivot_df = plot_df.pivot_table(
    index="group", columns="survived", values="count", fill_value=0
)

pivot_df.columns = ["Died", "Survived"]

# stacked bar chart
pivot_df.plot(kind="bar", stacked=True, figsize=(10, 6), color=["gray", "skyblue"])

plt.title("Titanic survival by passenger class and sex")
plt.xlabel("Passenger class + sex")
plt.ylabel("Number of passengers")
plt.xticks(rotation=45)
plt.legend(title="Outcome")

plt.tight_layout()
plt.show()

# ~~~~~ 2. Deviance comparison ~~~~~

deviances = pd.DataFrame(
    {
        "Model": ["Independence", "Full interaction"],
        "Deviance": [model_ind.deviance, model_inter.deviance],
    }
)

plt.figure(figsize=(7, 5))

bars = plt.bar(deviances["Model"], deviances["Deviance"])

plt.title("Model deviance comparison")
plt.ylabel("Deviance")


for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.show()

# ~~~~~ 3. Survival percentage by sex + class ~~~~~

perc_df = df.groupby(["passenger_class", "sex"])["survived"].mean().reset_index()

perc_df["survival_pct"] = perc_df["survived"] * 100


perc_df["group"] = perc_df["passenger_class"].astype(str) + " | " + perc_df["sex"]

plt.figure(figsize=(10, 6))

plt.bar(perc_df["group"], perc_df["survival_pct"], color="steelblue")

plt.title("Survival percentage by passenger class and sex")
plt.xlabel("Passenger class + sex")
plt.ylabel("Survival rate (%)")
plt.ylim(0, 100)

for i, v in enumerate(perc_df["survival_pct"]):
    plt.text(i, v + 2, f"{v:.1f}%", ha="center")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
