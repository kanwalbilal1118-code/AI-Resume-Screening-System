import pandas as pd

df = pd.read_csv("jobs_processed.csv")

small_df = df[
    [
        "Job Title",
        "Role",
        "Company",
        "combined_text"
    ]
]

small_df = small_df.sample(
    n=10000,
    random_state=42
)

small_df.to_csv(
    "jobs_small.csv",
    index=False
)

print("jobs_small.csv created")
print(small_df.shape)