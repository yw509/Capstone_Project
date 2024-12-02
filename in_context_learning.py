import os
import json
from pathlib import Path
from string import Template
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sklearn.metrics import f1_score

import sys
HOME_DIR = os.path.dirname(os.getcwd())
sys.path.append(HOME_DIR)
from utils import *

# backend infrastructure
domain = 'Taboola'
assert domain in ['Taboola', 'UCLANLP']
print('Setting up the openai backend with domain', domain)
if domain == 'Taboola':
    client = AzureOpenAI(
        azure_endpoint = "https://qa-la-002.openai.azure.com/",
        api_key=open('api_key_taboola.txt').readline().strip(),
        api_version="2024-02-15-preview"
    )
    rules_generation_model = "gpt-4-turbo-2024-04-09-research"  # 'dev-openai-gpt4-vision-preview' # dev-openai-o1-preview       # Model for rules generation
    image_check_model = "gpt-4o-2024-08-06-research"  # 'dev-azure-gpt4o'     # Model for image availability checking
    evaluation_model = "gpt-4o-2024-08-06-research"  # Model for evaluation
else:
    assert domain == 'UCLANLP'
    client = OpenAI(
        organization=open('organization_uclanlp.txt').readline().strip(),
        api_key=open('api_key_uclanlp.txt').readline().strip(),
        # base_url=openai_api_base,
    )
    rules_generation_model = 'gpt-4-turbo-2024-04-09'
    image_check_model = "gpt-4o-2024-08-06"
    evaluation_model = "gpt-4o-2024-08-06"
    # raise NotImplementedError


#template for in_context_learning prompt
#1. context
#2. task
#3. instruction
    #a. data analysis: the related cases from training dataset
    #b. inference: infer from the related cases
#4.output: approved/rejected
in_context_learning_prompt = Template("""
**Context:**
You are a very senior Ad Reviewer with twenty years experience. Your task is to thoroughly review all relevant information pertaining to an advertisement and determine whether it meets the criteria for approval or should be rejected.

**Task:**
For each assignment, you will be given a target advertisement that includes the title, description, and thumbnail image.
You will also receive 20 example ad components, each including a thumbnail image, title, description, approval_label, policy_domain, predicted_rejection_reason, predicted_sub_rejection, and predicted_rejection_comment. These ads have titles and descriptions similar to the one you are reviewing. Use these 20 ads as references to guide your decision on whether to approve or reject the ad you're working on. Note that if an ad's approval label is 'approved,' the predicted_rejection_reason, predicted_sub_rejection, and predicted_rejection_comment fields will be (null).

**Instruction:**

1. Data Analysis:
Analyze the provided dataset of example ads to become familiar with, understand, and fully grasp the reasons behind each ad's rejection or approval, identifying any underlying patterns.
Highlight contextual nuances that may affect approval decisions.

2. Label generation
Based on your analysis and what you have learned, carefully review the thumbnail image, title, and description of the target advertisement. Decide whether the target ad should be approved or rejected. If you decide to reject it, please provide the reason for rejection.

**Policy Domain:**
id: ${policy_domain_id}
name: ${policy_domain}

**Data Format**:

- **Target Advertisement**:
  - Image URL: ${target_image_url}
  - Title: ${target_title}
  - Description: ${target_description}
  - Policy Domain: ${target_policy_domain}

- **Example Ads**:
${example_ads_content}

**Output**:
Please provide your output in the following format:
Approval Label: {rejected/approved}
Predicted Rejection Comment: {(Provide the reason if rejected; enter (null) if approved)}

**Do not provide any string other than the approval label and predicted rejection comment.**
""")

def get_unique_top_n(bm25_model, eval_content, data, similar_item, remove_duplicate_titles):
    top_items = bm25_model.get_top_n(eval_content, data.to_dict('records'), n=len(data))
    
    if remove_duplicate_titles:
        title_counts = {}
        for ad in top_items:
            title_counts[ad['title']] = title_counts.get(ad['title'], 0) + 1
        
        unique_top_items = []
        for ad in top_items:
            if title_counts[ad['title']] == 1:  # Only add ads with unique titles
                unique_top_items.append(ad)
                if len(unique_top_items) == similar_item:  # Stop when we have the required number
                    break
        
        return unique_top_items
    else:
        return top_items[:similar_item]

def rank_bm25_retrieve(policy_domain_idx, policy_domain, eval_data_chunk, train_data_chunk, evaluation_model, similar_item, output_dir, remove_duplicate_titles, exp_name = None):
    y_true = []
    y_pred = []
    detailed_results = []  # List to store detailed results for each eval data point

    if not exp_name:
        print(f'Inference for policy domain: {policy_domain}')

    # Separate approved and rejected data and tokenize content
    approved_data = train_data_chunk[train_data_chunk['approval_label'] == 'approved']
    rejected_data = train_data_chunk[train_data_chunk['approval_label'] == 'rejected']
    approved_data['tokenized_content'] = approved_data.apply(lambda x: (str(x['title']) + ' ' + str(x['description'])).split(), axis=1)
    rejected_data['tokenized_content'] = rejected_data.apply(lambda x: (str(x['title']) + ' ' + str(x['description'])).split(), axis=1)
    
    # Initialize BM25 for approved and rejected datasets
    bm25_approved = BM25Okapi(approved_data['tokenized_content'])
    bm25_rejected = BM25Okapi(rejected_data['tokenized_content'])

    # For each evaluation data point, retrieve 10 most similar from approved and rejected train_data
    for _, eval_row in tqdm(eval_data_chunk.iterrows(), desc='Evaluating' if not exp_name else exp_name):
        eval_content = (str(eval_row['title']) + ' ' + str(eval_row['description'])).split()  # Tokenize eval data content
        
        # # Get top n similar approved and rejected items
        # top_approved = bm25_approved.get_top_n(eval_content, approved_data.to_dict('records'), n=similar_item)
        # top_rejected = bm25_rejected.get_top_n(eval_content, rejected_data.to_dict('records'), n=similar_item)
        # selected_train_items = top_approved + top_rejected

        top_approved = get_unique_top_n(bm25_approved, eval_content, approved_data, similar_item, remove_duplicate_titles)
        top_rejected = get_unique_top_n(bm25_rejected, eval_content, rejected_data, similar_item, remove_duplicate_titles)

        #print("The current number of examples is: (1) approved: ", len(top_approved), "; (2) rejected: ", len(top_rejected))
        # Combine approved and rejected items
        selected_train_items = top_approved + top_rejected  

        # render the prompt
        # fill the target advertisement data
        eval_content_dict = {
            "policy_domain_id": policy_domain_idx,
            "policy_domain": policy_domain,
            "target_image_url": eval_row['thumbnail_url'],
            "target_title": eval_row['title'],
            "target_description": eval_row['description'],
            "target_policy_domain": policy_domain
        }

        #prepare example advertisement data
        example_ads_list = []
        for ad in selected_train_items:
            ad_template = f"""
            - Image URL: {ad['thumbnail_url']}
            - Title: {ad['title']}
            - Description: {ad['description']}
            - Approval Label: {ad['approval_label']}
            - Policy Domain: {policy_domain}
            - Predicted Rejection Reason: {ad['predicted_rejection_reason']}
            - Predicted Sub-Rejection: {ad['predicted_sub_rejection']}
            - Predicted Rejection Comment: {ad['predicted_rejection_comment']}
            """
            example_ads_list.append(ad_template.strip())

        # Join example ads into a single string
        example_ads_content = "\n\n".join(example_ads_list)

        rendered_prompt = in_context_learning_prompt.substitute(
            policy_domain_id=eval_content_dict["policy_domain_id"],
            policy_domain=eval_content_dict["policy_domain"],
            target_image_url=eval_content_dict["target_image_url"],
            target_title=eval_content_dict["target_title"],
            target_description=eval_content_dict["target_description"],
            target_policy_domain=eval_content_dict["target_policy_domain"],
            example_ads_content=example_ads_content
        )

        messages = [
            {
                "role": "system",
                "content": rendered_prompt  # System-level prompt
            }
        ]

        kwargs = {
            "model": evaluation_model,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.5,
        }
        
        # get respond
        response = call_llm_gateway('uclanlp', kwargs)

        # Sample response from OpenAI API
        response_content = response.choices[0].message.content.strip('`json\n')

        # Extract the Approval Label from the response content
        # Assuming the response content format: "Approval Label: {label}\nPredicted Rejection Comment: {comment}"
        predicted_approval_label = None
        predicted_rejection_comment = None
        for line in response_content.splitlines():
            if "Approval Label:" in line:
                predicted_approval_label = line.split("Approval Label:")[1].strip().lower()  # Convert to lowercase and strip any surrounding whitespace
                # Clean any additional characters or punctuation
                predicted_approval_label = predicted_approval_label.replace(" ", "").replace(".", "").replace(",", "")
            elif "Predicted Rejection Comment:" in line:
                predicted_rejection_comment = line.split("Predicted Rejection Comment:")[1].strip()

        if predicted_approval_label not in ["rejected", "approved"]:
            print("Error: Approval label is not in the expected format.")
        # Now compare with the actual label
        else:
            y_true.append(eval_row['approval_label'].lower())  # modified: Store true label for F1 calculation
            y_pred.append(predicted_approval_label)

            # Append each example ad as a separate row in the detailed results
            for ad in selected_train_items:
                detailed_results.append({
                    "policy_domain": policy_domain,
                    "target_id": eval_row['id'],
                    "target_item_id": eval_row['item_id'],
                    "target_image_url": eval_row['thumbnail_url'],
                    "target_title": eval_row['title'],
                    "target_description": eval_row['description'],
                    "target_real_approval_label": eval_row['approval_label'].lower(),
                    "target_predicted_approval_label": predicted_approval_label,
                    "target_predicted_rejection_comment": predicted_rejection_comment,
                    "example_id": ad['id'],
                    "example_item_id": ad['item_id'],
                    "example_image_url": ad['thumbnail_url'],
                    "example_title": ad['title'],
                    "example_description": ad['description'],
                    "example_approval_label": ad['approval_label'].lower(),
                    "example_predicted_rejection_reason": ad['predicted_rejection_reason'],
                    "example_predicted_sub_rejection": ad['predicted_sub_rejection'],
                    "example_predicted_rejection_comment": ad['predicted_rejection_comment']
                })
    
    f1 = f1_score(y_true, y_pred, average='macro')

    # Convert detailed results to DataFrame and save in specified directory
    detailed_results_df = pd.DataFrame(detailed_results)
    output_file = os.path.join(output_dir, f"{policy_domain.replace(' ', '_')}_similar_item_{similar_item}_evaluation_results.csv")
    detailed_results_df.to_csv(output_file, index=False)
    print(f"Saved detailed results to {output_file}")

    return f1


def main(model):

    base_dir = "/Users/wangyuchen/Desktop/research_with_Taboola/abby-taboola/meta_prompting/single_iter_meta_prompt/in_context_learning_nosametitles_norules_results"

    policy_domains = ['Copyrights and Competitive Claims', 'Endorsement', 'Exploitative', 'Finance Claims', 'Health Claims',
                      'Misrepresentative', 'Offensive', 'Politicized', 'Quality', 'Sexualized or Skin']
    policy_domain_to_idx = {x: i+1 for i, x in enumerate(policy_domains)}
    
    # read data
    data = json.load(open(os.path.join(HOME_DIR, 'data/1004_meta_prompting_promotional_content_personal_finance_investing_only_dataset_splitted.json')))
    data_df = {}
    for k, v in data.items():
        # print(v['train'])
        # print(type(v['train']))
        data_df[k] = {
            'train_df': pd.read_json(json.dumps(v['train'])),
            'eval_df':  pd.read_json(json.dumps(v['eval']))
        }

    similar_items = [80, 90, 100]

    results = []
    for similar_item in similar_items:
        # Create directory for each similar_item
        similar_item_dir = os.path.join(base_dir, f"similar_item_{similar_item}")
        os.makedirs(similar_item_dir, exist_ok=True)

        for policy_domain in policy_domains:
            policy_domain_idx = policy_domain_to_idx[policy_domain]
            f1_score = rank_bm25_retrieve(policy_domain_idx, policy_domain, data_df[policy_domain]['eval_df'], data_df[policy_domain]['train_df'], model, similar_item, similar_item_dir, True) 
            print(f"The macro F1-score for {policy_domain} with similar_item={similar_item} is: {f1_score:.4f}")

            # Append results to the list
            results.append({
                "policy_domain": policy_domain,
                "similar_item": similar_item,
                "macro_f1_score": f1_score
            })

    # Create a DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    # Save the DataFrame to CSV
    results_df.to_csv("rank_bm25_nosametitles_macro_f1_scores_higher_n.csv", index=False)

    # Print out the absolute path of the CSV file
    csv_path = os.path.abspath("rank_bm25_nosametitles_macro_f1_scores_higher_n.csv")
    print(f"Results saved to {csv_path}")
    

if __name__ == '__main__':
    evaluation_model = "gpt-4o-2024-08-06"
    main(evaluation_model)
