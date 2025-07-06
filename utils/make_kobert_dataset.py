import pandas as pd
import glob
import os

# 1: merge all CSV files in the directory
csv_dir = "resume_labeling_data"
csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# 2: create input_text column
df_all["input_text"] = "[" + df_all["cat"] + "] [" + df_all["subcat"] + "] " + df_all["text"]

# 3: map grade to label
grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
df_all["label"] = df_all["grade"].map(grade_map)

# 4: save to CSV for KoBERT training
df_all.to_csv("kobert_dataset.csv", index=False)

print("Finished: kobert_dataset.csv")
