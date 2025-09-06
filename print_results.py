import pandas as pd

dataset = "churn"
generator = "xgenboost"
file_name = f"results/{dataset}_{generator}.csv"

results = pd.read_csv(file_name)
results = results.groupby("metric").mean(numeric_only=True)
results.to_csv(f"results/{dataset}_{generator}_aggregated.csv")
print(results)
