import pandas as pd

jobs = pd.read_csv("../dataset/job_descriptions.csv")

jobs["combined_text"] = (
    jobs["Job Description"].fillna("") + " " +
    jobs["skills"].fillna("") + " " +
    jobs["Responsibilities"].fillna("")
)

jobs.to_csv("jobs_processed.csv", index=False)

print("jobs_processed.csv created successfully")