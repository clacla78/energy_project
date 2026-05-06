import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
# Using your filename: owners_coowners_income.csv
file_path = "owners_coowners_income.csv"
df = pd.read_csv(file_path)

# 2. Function to parse and calculate the midpoint
def clean_income(income_str):
    try:
        clean_s = str(income_str).replace("'", "").replace(" CHF", "")
        parts = clean_s.split(" - ")
        return (float(parts[0]) + float(parts[1])) / 2
    except:
        return np.nan

# Process the source data (Lausanne)
df['lausanne_income'] = df['income'].apply(clean_income)
df = df.dropna(subset=['lausanne_income'])

# 3. Translation to St-Prex
target_mean_st_prex = 110493.0
current_mean_lausanne = df['lausanne_income'].mean()
scaling_factor = target_mean_st_prex / current_mean_lausanne

# Create the translated column
df['st_prex_income'] = df['lausanne_income'] * scaling_factor

# 4. Print Summary Stats
print(f"--- Geographical Translation Summary ---")
print(f"Source (Lausanne) Average: {current_mean_lausanne:,.2f} CHF")
print(f"Target (St-Prex) Average:  {target_mean_st_prex:,.2f} CHF")
print(f"Scaling Multiplier:       {scaling_factor:.4f}")

# 5. Visualization
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Histogram/Density Repartition
sns.histplot(df['lausanne_income'], kde=True, color="#3498db", label="Source: Lausanne", ax=axes[0], alpha=0.5)
sns.histplot(df['st_prex_income'], kde=True, color="#e67e22", label="Target: St-Prex", ax=axes[0], alpha=0.5)
axes[0].set_title("Salary Repartition: Lausanne vs. St-Prex", fontweight='bold')
axes[0].set_xlabel("Annual Income (CHF)")
axes[0].legend()

# Plot 2: Boxplot for Spread, Quantiles, and Extremes
# Format data for seaborn
melted_df = df[['lausanne_income', 'st_prex_income']].melt(var_name='Location', value_name='Income')
melted_df['Location'] = melted_df['Location'].replace({
    'lausanne_income': 'Lausanne (Source)', 
    'st_prex_income': 'St-Prex (Scaled)'
})

sns.boxplot(x='Location', y='Income', data=melted_df, palette=['#3498db', '#e67e22'], ax=axes[1])
axes[1].set_title("Spread and Extreme Comparisons", fontweight='bold')
axes[1].set_ylabel("Income (CHF)")

plt.tight_layout()
plt.show()