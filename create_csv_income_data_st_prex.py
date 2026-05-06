import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and Clean Source Data (Lausanne)
file_path = "owners_coowners_income.csv"
df_orig = pd.read_csv(file_path)

def clean_income(income_str):
    try:
        # Removing formatting characters and splitting the range
        clean_s = str(income_str).replace("'", "").replace(" CHF", "")
        parts = clean_s.split(" - ")
        return (float(parts[0]) + float(parts[1])) / 2
    except:
        return np.nan

# Extract valid midpoint salaries from the source file
lausanne_incomes = df_orig['income'].apply(clean_income).dropna().values

# 2. Calculate Scaling Factor for St-Prex
target_mean = 110493.0
current_mean = np.mean(lausanne_incomes)
scaling_factor = target_mean / current_mean
st_prex_theoretical = lausanne_incomes * scaling_factor

# 3. Generate 536 Agents (Bootstrap Sampling)
num_agents = 536
# We pick 536 samples randomly from the theoretical distribution
generated_salaries = np.random.choice(st_prex_theoretical, size=num_agents, replace=True)

# --- MEAN CORRECTION ---
# Because random sampling can slightly shift the mean (e.g., to 113k),
# we apply a mathematical correction to lock it at exactly 110,493 CHF.
actual_gen_mean = np.mean(generated_salaries)
correction_ratio = target_mean / actual_gen_mean
final_salaries = generated_salaries * correction_ratio

# 4. Create and Export the CSV
df_agents = pd.DataFrame({
    'agent_id': range(1, num_agents + 1),
    'location': 'St-Prex',
    'salary_chf': np.round(final_salaries, 2)
})

# Shuffle to ensure no ordering bias
df_agents = df_agents.sample(frac=1).reset_index(drop=True)
df_agents.to_csv("st_prex_agents_536.csv", index=False)

print(f"Target Mean: {target_mean:,.2f} CHF")
print(f"Final Generated Mean: {df_agents['salary_chf'].mean():,.2f} CHF")

# 5. Distribution Visualization
plt.figure(figsize=(10, 6))

# Theoretical Distribution (The "Goal" shape)
sns.kdeplot(st_prex_theoretical, label="Theoretical Distribution (St-Prex)", 
            color="blue", bw_adjust=1, fill=True, alpha=0.3)

# Actual Generated Data (The 536 agents)
sns.kdeplot(df_agents['salary_chf'], label="Generated 536 Agents", 
            color="red", bw_adjust=1, linestyle="--")

plt.title("Comparison: Theoretical St-Prex Distribution vs. Generated Agents")
plt.xlabel("Salary (CHF)")
plt.ylabel("Density")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()