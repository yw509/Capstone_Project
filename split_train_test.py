import json
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dataset_path = '1004_meta_prompting_promotional_content_personal_finance_investing_only_dataset.csv'

    default_random_state = 42
    default_test_size = 0.4
    max_test_sample_count = 100

    data_file_path = dataset_path
    df = pd.read_csv(data_file_path)
    print("Dataset shape:", df.shape)
    print(df.head())

    # Map approval status to binary labels
    df['approval_label'] = df['predicted_component_approval_status'].map({
        'SAFE': 'approved',
        'LOW_BROW': None,
        'REJECTED': 'rejected'
    })
    
    # Verify the mapping
    print(df['approval_label'].value_counts(dropna=False))    
    
    df[df.approval_label == 'approved'].predicted_rejection_comment.value_counts(dropna=False)
    # clean invalid data (should be done once and saved)
    df = df[~df.approval_label.isna()]

    print(len(df))

    df.drop_duplicates(subset=['item_id'], inplace=True, ignore_index=True)
    df[df.approval_label == 'rejected'].relevant_component.value_counts(dropna=False)

    # Create group datasets by policy_domain
    df_policy_domain_groups = df.groupby('policy_domain')
    df_approved = df[df['approval_label'] == 'approved']
    # Print the number of ads in each policy domain
    df_policy_domain_datasets = {}
    for policy_domain, df_policy_domain in df_policy_domain_groups:
        if policy_domain in ['(null)', 'Safety Setting']:  # Skip null policy domain (safe items) and Safety Setting domain
            continue
        df_approved_sample = df_approved.sample(n=df_policy_domain.shape[0], random_state=default_random_state)
        df_policy_domain_with_approve = pd.concat([df_policy_domain, df_approved_sample])
        test_size = min(default_test_size, max_test_sample_count / df_policy_domain_with_approve.shape[0])
        train_df, eval_df = train_test_split(df_policy_domain_with_approve, test_size=test_size, random_state=default_random_state, stratify=df_policy_domain_with_approve['approval_label'])
        df_policy_domain_datasets[policy_domain] = {
            'train': [json.loads(x.strip()) for x in train_df.to_json(orient='records', lines=True).split('\n') if x.strip()],
            'eval': [json.loads(x.strip()) for x in eval_df.to_json(orient='records', lines=True).split('\n') if x.strip()],
        }
        print(f'Policy Domain: {policy_domain}:\n', f'train_df size: {train_df.shape[0]}, eval_df size: {eval_df.shape[0]}') #, avoid_overfitting_df size: {avoid_overfitting_df.shape[0]}')

    out_dataset_path = dataset_path.replace('.csv', '_splitted.json')
    assert out_dataset_path != dataset_path
    json.dump(df_policy_domain_datasets, open(out_dataset_path, 'w'), indent=4)
    print('Saved to', out_dataset_path)
    