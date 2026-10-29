import pandas as pd
import numpy as np
import os

np.random.seed(42)
n_samples = 1000

cgpa = np.random.normal(7.2, 1.1, n_samples).clip(4.0, 10.0)
internships = np.random.poisson(1.2, n_samples).clip(0, 5)
technical_score = np.random.normal(70, 15, n_samples).clip(30, 100)
soft_skills = np.random.randint(1, 11, n_samples)
gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.55, 0.45])

# Vectorized check for gender bias
gender_bias = np.where(gender == "Male", 1.2, -0.8)

logit = (
    0.8 * cgpa 
    + 0.5 * internships 
    + 0.05 * technical_score 
    + 0.3 * soft_skills 
    + gender_bias
    - 10.0
)
prob = 1 / (1 + np.exp(-logit))
placed = (prob > 0.5).astype(int)

df = pd.DataFrame({
    "CGPA": np.round(cgpa, 2),
    "Internships": internships,
    "TechnicalScore": np.round(technical_score, 1),
    "SoftSkills": soft_skills,
    "Gender": gender,
    "Placed": placed
})

os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/placement_data.csv", index=False)
print("Dataset created successfully at dataset/placement_data.csv!")