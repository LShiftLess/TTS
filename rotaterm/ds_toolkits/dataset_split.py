import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", "--dataset_dir", dest="dataset_dir", required=True)
    parser.add_argument("--test_size", dest="test_size", type=float, default=0.1)
    parser.add_argument("--random_state", dest="random_state", type=int, default=42)
    args = parser.parse_args()

    ds_dir = args.dataset_dir
    test_size = args.test_size
    random_s = args.random_state

    if not 'metadata.csv' in os.listdir(ds_dir):
        raise FileNotFoundError("There is no file named 'metadata.cvs' in the dataset folder!")

    # Load metadata.csv
    df = pd.read_csv(os.path.join(ds_dir, "metadata.csv"), sep="|", header=None, names=["file_name", "transcription", "normalized"])

    # Split into train and validation
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_s)

    # Save to separate files
    train_df.to_csv(os.path.join(ds_dir, "metadata.train.csv"), sep="|", header=False, index=False)
    val_df.to_csv(os.path.join(ds_dir, "metadata.val.csv"), sep="|", header=False, index=False)
    
if __name__ == "__main__":
    main()