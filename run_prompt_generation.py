import os
import json
from pathlib import Path
from jinja2 import Template
import pandas as pd
from datetime import datetime

import sys
HOME_DIR = os.path.dirname(os.getcwd())
sys.path.append(HOME_DIR)
from utils import *

    
initial_rule_generation_prompt = """
**Context:**
You are a very senior policy writer with twenty years experience. Your task is generating a detailed set of policy compliance rules for reviewing advertisements. The rules you generated will be give to your AI colleague, GPT-4o, to review the ads. So please try your best to generate the most accurate and clear rules to help your colleague to ensure the ads comply with the policy.

**Task:**
You will be given a specific policy domain.
You will be also provided with a dataset of ads components (title, description, thumbnail image, etc) that are grouped into compliant and non-compliant examples related to the policy domain.

**Instruction:**

1. Data Analysis:
Analyze the provided dataset of compliant and non-compliant ads to infer the underlying rules governing ad approval.
Highlight contextual nuances that may affect approval decisions.

2. Rule generation
Generate a set of policy rules, including:
  - An overall rule to briefly describe this policy domain.
  - A set of sub-rules to cover different aspects of the policy domain.
When you generate the rule, consider the following:
  - Generate the rule based on your analysis report.
  - Emphasize clear communication by stating rules in precise and unambiguous language.
  - Derive clear and concise rules based on consistent patterns in the dataset.
  - Implement conditional statements and utilize sub-rules to capture exceptions and nuances.
  - State the rule clearly and concisely, using conditional language if appropriate (e.g., "Ads must not... unless...").

**Policy Domain:**
id: {{policy_domain_id}}
name: {{policy_domain}}

**Output**:
Provide your entire response in JSON format.
Use the following structure:
{
  "policy_domain_id": "policy domain id",
  "policy_domain": "policy domain name",
  "analysis_report": [
    {
      "underlying_rule": "Description of an underlying rule you found.",
      "highlight_nuances": "The highlight contextual nuances you found in the ads components (text,image,etc)."
    },
    ...
  ],
  "policy_domain_description": "Briefly describe the rule of this policy domain.",
  "policy_rules": [
    {
      "id": "The id format is: {policy_domain_id}.{rule_id(auto incremental)}, e,g 1.1, 1.2",
      "rule": "State the rule clearly and concisely."
  ]
}
**Do not provide any string other than those inside the key value pairs.**
"""


def main(seed):
    # data chunk size
    default_approved_init_size = 30
    default_rejected_init_size = 70
    final_approved_max_examples = 30  # 6
    final_rejected_max_examples = 70  # 14
    
    #seed = 2024   # 42 1234 2024
    img_detail_level = 'low'

    policy_domains = ['Copyrights and Competitive Claims', 'Endorsement', 'Exploitative', 'Finance Claims', 'Health Claims', 'Misrepresentative', 'Offensive', 'Politicized', 'Quality', 'Sexualized or Skin']

    # read data
    data = json.load(open(os.path.join(HOME_DIR, 'data/1004_meta_prompting_promotional_content_personal_finance_investing_only_dataset_splitted.json')))
    data_df = {}
    for k, v in data.items():
        data_df[k] = {
            'train_df': pd.read_json(json.dumps(v['train'])),
            'eval_df':  pd.read_json(json.dumps(v['eval']))
        }
        #print(v.keys())
        #print(len(data_df[k]['train_df']))
        #print(len(data_df[k]['eval_df']))
            
    # output directories
    exp = 'single_iter_meta_prompt'
    parent_dir = f'results/{exp}/{final_approved_max_examples}approved_{final_rejected_max_examples}rejected_seed{seed}' #_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    rule_output_dir = os.path.join(HOME_DIR, parent_dir, 'rules')
    evaluation_output_dir = os.path.join(HOME_DIR, parent_dir, 'evaluation')
    visualization_output_dir = os.path.join(HOME_DIR, parent_dir, 'visualization')
    Path(rule_output_dir).mkdir(parents=True, exist_ok=True)
    Path(evaluation_output_dir).mkdir(parents=True, exist_ok=True)
    Path(visualization_output_dir).mkdir(parents=True, exist_ok=True)

    # generate prompts
    policy_domain_to_idx = {x: i+1 for i, x in enumerate(policy_domains)}
    for policy_domain in policy_domains:
        df_policy_domain_data = data_df[policy_domain]
        policy_domain_idx = policy_domain_to_idx[policy_domain]
        train_df = df_policy_domain_data['train_df']
        
        print('\n\n=========================================\n\n')
        print(f'Single-iteration rule generation for policy domain: {policy_domain_idx}. {policy_domain}')
        
        # In-context examples
        approved_data_chunk = get_data_chunk(train_df[train_df['approval_label'] == 'approved'], default_approved_init_size, random_state=seed)
        rejected_data_chunk = get_data_chunk(train_df[train_df['approval_label'] == 'rejected'], default_rejected_init_size, random_state=seed)
        approved_data_chunk, approved_data_valid_llm_responses = get_data_chunk_with_valid_thumbnail_url(approved_data_chunk, img_detail_level=img_detail_level)
        rejected_data_chunk, rejected_data_valid_llm_responses = get_data_chunk_with_valid_thumbnail_url(rejected_data_chunk, img_detail_level=img_detail_level)

        print('Before thumbnail check:', len(approved_data_chunk), len(rejected_data_chunk))
        print('After thumbnail check:', len(approved_data_chunk), len(rejected_data_chunk))
        print(f'Warning: truncating examples to {default_approved_init_size} approved and {default_rejected_init_size} rejected instances')
        
        label_to_data_chunk_dict = {
            'Compliant Ads': approved_data_chunk[:final_approved_max_examples],
            'Non-Compliant Ads': rejected_data_chunk[:final_rejected_max_examples],
        }
        
        # render the system prompt
        system_prompt = Template(initial_rule_generation_prompt).render(policy_domain_id=policy_domain_idx, policy_domain=policy_domain)
        
        # Convert the dataset to user prompt
        user_prompt = convert_dataset_to_user_prompt(label_to_data_chunk_dict, components_override=None, is_relevant_component_only=True,
                                                     included_comment_col='predicted_rejection_comment', img_detail_level=img_detail_level)
        
        # Build the LLM Gateway request
        llm_request = build_llm_gateway_request(system_prompt=system_prompt, user_prompt=user_prompt, model=rules_generation_model,
                                                max_tokens=1000, temperature=0.5, seed=seed)
        
        response = call_llm_gateway('uclanlp', llm_request)
        
        print(response.choices[0].message.content.strip('`json\n'))
        
        # Save the response
        response_file_path = os.path.join(rule_output_dir, f'{policy_domain}_initial_rule_generation_response.json')
        with open(response_file_path, 'w') as f:
            json.dump(json.loads(response.choices[0].message.content.strip('`json\n')), f, indent=4)
            print(f'Initial rule generation response saved to: {response_file_path}')
            

if __name__ == '__main__':
    for seed in range(107, 111):
        main(seed)
