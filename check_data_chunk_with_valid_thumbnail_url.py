import os
import json
import pandas as pd

import sys
HOME_DIR = os.path.dirname(os.getcwd())
sys.path.append(HOME_DIR)
from utils import *

def preprocess_data_and_save(output_path):
    # Read data
    data = json.load(open(os.path.join(HOME_DIR, 'data/1004_meta_prompting_promotional_content_personal_finance_investing_only_dataset_splitted.json')))
    data_df = {}

    # Pre-check for valid thumbnail URLs
    print("Checking for valid thumbnail URLs in train and eval datasets...")
    for k, v in data.items():
        print(f"Processing policy domain: {k}")

        # Process train data
        train_df = pd.read_json(json.dumps(v['train']))
        valid_train_df, train_valid_responses = get_data_chunk_with_valid_thumbnail_url(train_df, "low")
        
        # Process eval data
        eval_df = pd.read_json(json.dumps(v['eval']))
        valid_eval_df, eval_valid_responses = get_data_chunk_with_valid_thumbnail_url(eval_df, "low")
        
        # Store valid datasets
        data_df[k] = {
            'train_df': valid_train_df,
            'eval_df': valid_eval_df
        }

    # Save the data_df to a pickle file
    pd.to_pickle(data_df, output_path)
    print(f"Filtered data has been saved to {output_path}")

if __name__ == "__main__":
    output_file = os.path.join(HOME_DIR, 'data/filtered_1004_meta_data_with_valid_thumbnail_url.pkl')  # Change filename as needed
    preprocess_data_and_save(output_file)
