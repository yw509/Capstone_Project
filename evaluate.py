# evaluation file for meta-prompting

import os
import json
from pathlib import Path
from jinja2 import Template
import pandas as pd
from tqdm import tqdm

import sys
HOME_DIR = os.getcwd()
sys.path.append(HOME_DIR)
from utils import *


def evaluate_single_policy_domain(policy_domain_idx, policy_domain, policy_domain_rules,
                                  test_data_chunk, evaluation_model, exp_name=None):
    if not exp_name:
        print(f'Inference for policy domain: {policy_domain}')

    eval_system_prompt = Template(evaluation_prompt).render(policy_domains=[(f'{policy_domain_idx}. {policy_domain}', pd.DataFrame.from_dict(policy_domain_rules))])

    refinement_inference_data_user_prompts = [convert_dataset_to_user_prompt({'Evaluation Ads': row.to_frame().T},
                                                                             components_override=['item_id','thumbnail_url','title','description'])
                                              for _, row in test_data_chunk.iterrows()]

    # Build the LLM Gateway request for inference refinement
    refinement_inference_data_llm_requests = [build_llm_gateway_request(system_prompt=eval_system_prompt, user_prompt=user_prompt, model=evaluation_model,
                                                                        max_tokens=1000, temperature=0.0, seed=42)
                                              for user_prompt in refinement_inference_data_user_prompts]

    refinement_inference_responses = []
    for r in tqdm(refinement_inference_data_llm_requests, desc='Evaluating' if not exp_name else exp_name):
        try:
            refinement_inference_responses.append(call_llm_gateway('taboola', r))
        except:
            continue
        
    # convert eval responses to dataframe
    refinement_inference_results = []
    for response in refinement_inference_responses:
        try:
            json_response = json.loads(response.choices[0].message.content.replace('```json', '').replace('```', '').strip())
            refinement_inference_results.append(json_response.get('ad_evaluations')[0])
        except Exception as e:
            continue

    refinement_inference_results_df = pd.DataFrame(refinement_inference_results)
    refinement_inference_results_df['ad_id'] = refinement_inference_results_df['ad_id'].astype(int)
    
    df_refinement_inference_merged = pd.merge(test_data_chunk, refinement_inference_results_df, left_on='item_id', right_on='ad_id')
    df_refinement_inference_merged = df_refinement_inference_merged[df_refinement_inference_merged.compliance_status.notna()]
    return df_refinement_inference_merged


def main(working_dir, model):
    assert 'rules/' in working_dir 
    out_dir = working_dir.replace('rules/', 'evaluation/')

    policy_domains = ['Copyrights and Competitive Claims', 'Endorsement', 'Exploitative', 'Finance Claims', 'Health Claims',
                      'Misrepresentative', 'Offensive', 'Politicized', 'Quality', 'Sexualized or Skin']
    policy_domain_to_idx = {x: i+1 for i, x in enumerate(policy_domains)}
    
    # read data
    data = json.load(open(os.path.join(HOME_DIR, 'data/1004_meta_prompting_promotional_content_personal_finance_investing_only_dataset_splitted.json')))
    data_df = {}
    for k, v in data.items():
        data_df[k] = {
            'train_df': pd.read_json(json.dumps(v['train'])),
            'eval_df':  pd.read_json(json.dumps(v['eval']))
        }

    metrics_list = []
    for policy_domain in policy_domains:
        # for policy_domain in ['Finance Claims', 'Health Claims', 'Misrepresentative', 'Offensive', 'Politicized', 'Quality', 'Sexualized or Skin']:   
        # read policy output
        with open(os.path.join(working_dir, f'{policy_domain}_initial_rule_generation_response.json'), 'r') as f:
            current_policy_output = json.load(f)
            current_rules = current_policy_output['policy_rules']

        # inference
        policy_domain_idx = policy_domain_to_idx[policy_domain]
        df_eval_merged = evaluate_single_policy_domain(policy_domain_idx, policy_domain, current_rules,
                                                       data_df[policy_domain]['eval_df'], model)    
        df_eval_merged.to_csv(out_dir + '/detailed_results_{}.csv'.format(policy_domain))

        # calculate scores
        eval_report, eval_cm_df = calculate_eval_metric(df_eval_merged, is_print=False)

        print(json.dumps(eval_report, indent=4))
        f1 = eval_report.get('macro avg').get('f1-score')
        
        # save the outputs and scores
        metrics_list.append({
            'policy_domain': policy_domain,
            'iteration': 0,
            'eval_macro_avg_f1_score': eval_report.get('macro avg').get('f1-score'),
            'eval_report': eval_report,
            'eval_num_samples': df_eval_merged.shape[0],
            'eval_tp': eval_cm_df.loc['Actual Rejected', 'Predicted Rejected'].item(),
            'eval_tn': eval_cm_df.loc['Actual Approved', 'Predicted Approved'].item(),
            'eval_fp': eval_cm_df.loc['Actual Approved', 'Predicted Rejected'].item(),
            'eval_fn': eval_cm_df.loc['Actual Rejected', 'Predicted Approved'].item()
        })
        json.dump(metrics_list[-1], open(out_dir + '/metrics_{}.json'.format(policy_domain), 'w'), indent=4)
        
    json.dump(metrics_list, open(out_dir + '/metrics_all.json', 'w'), indent=4)
    print(out_dir + '/metrics_all.json')

if __name__ == '__main__':
    in_dir = '/home/diwu/research_with_Taboola/abby-taboola/meta_prompting/results/single_iter_meta_prompt/30approved_70rejected_seed2024/rules/'
    evaluation_model = "gpt-4o-2024-08-06-research"
    main(in_dir, evaluation_model)
