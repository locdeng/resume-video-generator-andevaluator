from sklearn.model_selection import train_test_split
import pandas as pd

# read CSV file
df = pd.read_csv("kobert_dataset.csv")

# split train (80%) and temp (20%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# split temp into val (10%) and test (10%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

# save 3 files
train_df.to_csv("kobert_train.csv", index=False)
val_df.to_csv("kobert_val.csv", index=False)
test_df.to_csv("kobert_test.csv", index=False)

print("Completely saved 3 train/val/test")
