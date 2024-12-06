# multi-iteration meta-prompting

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import json
from pathlib import Path
from jinja2 import Template
import numpy as np
import pandas as pd
from datetime import datetime
import random

import sys
HOME_DIR = os.path.dirname(os.getcwd())
sys.path.append(HOME_DIR)
from utils import *
from evaluate import evaluate_single_policy_domain

refinement_prompt = """
**Context:**
You are a very senior policy writer with twenty years experience. Your task is to refine the set of policy compliance rules for reviewing advertisements. The rules you generated will be give to your AI colleague, GPT-4o, to review the ads. So please try your best to generate the most accurate and clear rules to help your colleague to ensure the ads comply with the policy.

**Task:**
You will be provided a specific policy domain and a list of policy rules related to that domain.
You will also be provided with a dataset of ads components that has been reviewed by your AI colleague GPT-4o based on the policy rules provided above. The dataset is grouped into 4 categories: true positives, false positives, true negatives, and false negatives.
Based on the given data, the goal is to refine the policy rules to improve the accuracy of the ad review process.

**Instructions:**
1. **Data Analysis**:
Analyze the policy rules and your colleague's review result in the ads dataset, to understand why your colleague made correct/incorrect review result:
  A. Correct Review Results:
    - True Positives (Correctly Rejected): These ads were non-compliant and correctly rejected by your colleague based on the current policy rules.
    - True Negatives (Correctly Approved): These ads were compliant and correctly approved by your colleague based on the current policy rules.
  B. Incorrect Review Results:
    - False Positives (Incorrectly Rejected): These ads were compliant but incorrectly rejected by your colleague based on the current policy rules. Identify issues that led to these false alarms and suggest modifications on policy rules to avoid such errors.
    - False Negatives (Incorrectly Approved): These ads were non-compliant but incorrectly approved by your colleague based on the current policy rules. Determine what was missed and how the policy rules should be adjusted to reject these ads correctly.
Highlight contextual nuances that may affect approval decisions.
Describe your refinement strategy for each correct/incorrect review pattern you found. You can consider the following strategies:
  - Modify existing rules to capture more nuanced patterns.
  - Add new rules to cover exceptions or edge cases.
  - Remove incorrect rules.
  - Adjust the conflicted rules.
  - Keep this pattern when proposing new rule because it is correct.

2. **Propose a new rule set**:
Based on your analysis report, propose a new set of policy rules to improve the overall accuracy.
You should follow the same structure when you propose the new refined rule set. Here is a simple guidance of the structure:
  - A set of rules to cover different aspects of the policy domain.
When refining the rules, consider the following:
  - Generate the rule based on your refinement strategy, analysis report and the current policy rules.
  - Emphasize clear communication by stating rules in precise and unambiguous language.
  - Derive clear and concise rules based on consistent patterns in the dataset.
  - Implement conditional statements and utilize sub-rules to capture exceptions and nuances.
  - State the rule clearly and concisely, using conditional language if appropriate (e.g., "Ads must not... unless...").
  - Avoid to write rules which can be conflicted with other rules.

**Current Policy Rules:**
{% for policy_domain, rules in policy_domains %}
{{ policy_domain }}:
{% for _, rule in rules.iterrows() %}
{{ rule.id }}. {{ rule.rule }}
{% endfor %}
{% endfor %}

**Output**:
Provide your entire response in JSON format.
Use the following structure:
{
  "policy_domain_id": "policy domain id",
  "policy_domain": "policy domain name",
  "analysis_report": {
    "correct_review_patterns": [
      {
        "pattern_summary": "summary of this correct review pattern",
        "related_policy_rule_id": "", // the policy rule id related to this pattern from the current policy set, e.g. 1.1, 2.3, 3(all rules), etc.
        "related_policy_rule": "the policy rule related to this pattern from the current policy set",
        "highlight_nuances_or_patterns": "The highlight contextual nuances or patterns you found in the ads components (text,image,etc).",
        "refinement_strategy": "Briefly describe your refinement strategy for this pattern."
      },
      ...
    ],
    "incorrect_review_patterns": [
      {
        "pattern_summary": "summary of this incorrect review pattern",
        "related_policy_rule_id": "", // the policy rule id related to this pattern from the current policy set, e.g. 1.1, 2.3, 3(all rules), etc.
        "related_policy_rule": "the policy rule related to this pattern from the current policy set",
        "highlight_nuances_or_patterns": "The highlight contextual nuances or patterns you found in the ads components (text,image,etc).",
        "refinement_strategy": "Briefly describe your refinement strategy for this pattern."
      },
      ...
    ]
  },
  "policy_rules": [
    {
      "id": "The id format is: {policy_domain_id}.{rule_id(auto incremental)}, e,g 1.1, 1.2",
      "rule": "State the rule clearly and concisely."
    },
    ...
  ]
}
**Do not provide any string other than those inside the key value pairs.**
"""

eval_required_fields =['item_id','thumbnail_url','title','description']
refinement_required_fields = ['item_id','thumbnail_url','title','description','true_approval_label','your_colleague_approval_label']

def main(base_dir, rules_generation_model, prediction_model, working_policy_domains, seed, 
         example_pool_size=None, example_ranking_strategy=None, example_ranking_strategy_args={}):
    improvement_criterion = 'train'  # train trainvalid
    assert improvement_criterion in ['train', 'trainvalid']
    # seed = 2024   # 42 1234 2024
    random.seed(seed)
    print(f'seed{seed}')
    print(base_dir)
    img_detail_level = 'low'
    
    # parameters
    n_example_per_round_approved = 20
    n_example_per_round_rejected = 20
    max_iterations = 50
    exp = 'iterative_meta_prompt'
    parent_dir = f'{base_dir}/{exp}/{n_example_per_round_approved}approved_{n_example_per_round_rejected}rejected_seed{seed}_criterion{improvement_criterion}_{max_iterations}iters' #_{datetime.now().strftime("%Y%m%d-%H%M%S")}/'
    
    if example_pool_size:
        parent_dir += f'_expoolsize{example_pool_size}'
    if example_ranking_strategy:
        parent_dir += f'_exrankstrategy{example_ranking_strategy}'
    
    # read data
    data = json.load(open(os.path.join(HOME_DIR, 'data/1004_meta_prompting_promotional_content_personal_finance_investing_only_dataset_splitted_with_valid.json')))
    data_df = {}
    for k, v in data.items():
        data_df[k] = {
            'train_df': pd.read_json(json.dumps(v['train'])),
            'valid_df': pd.read_json(json.dumps(v['valid'])),
            'eval_df':  pd.read_json(json.dumps(v['eval']))
        }
        
    for policy_domain_idx, policy_domain in enumerate(working_policy_domains):
        print('Start prompt optimization for', policy_domain)
        
        cur_policy_metrics = []
        
        train_df = data_df[policy_domain]['train_df']
        valid_df = data_df[policy_domain]['valid_df']
        valid_data_chunk = valid_df
        eval_df = data_df[policy_domain]['eval_df']
        
        with open(os.path.join(base_dir, f'{policy_domain}_initial_rule_generation_response.json'), 'r') as f:
            current_policy_output = json.load(f)
            best_rules = current_policy_output['policy_rules']
            
        # eval before start
        eval_report_valid_before, _ = calculate_eval_metric(evaluate_single_policy_domain(policy_domain_idx+1, policy_domain, best_rules, 
                                                            valid_data_chunk, prediction_model, exp_name='Eval on valid, iter 0'), is_print=False)
        f1_valid_best = eval_report_valid_before.get('macro avg').get('f1-score')
        
        best_iter = 0
        valid_report_best_iter = eval_report_valid_before
        proposal_best_iter = best_rules
        
        for iter_id in range(1, max_iterations+1):
            print('\n\n\n==========================================\n\n\n')
            print(policy_domain, 'iteration', iter_id)
            cur_round_output_dir = os.path.join(base_dir, parent_dir, f'iter_{iter_id}/')
            Path(cur_round_output_dir).mkdir(parents=True, exist_ok=True)
            
            # 1 - sample training examples for refinement
            if not example_pool_size:
                approved_data_chunk = get_data_chunk(train_df[train_df['approval_label'] == 'approved'], n_example_per_round_approved, random_state=random.randint(1, 10000000))
                rejected_data_chunk = get_data_chunk(train_df[train_df['approval_label'] == 'rejected'], n_example_per_round_rejected, random_state=random.randint(1, 10000000))
            else:
                approved_data_chunk = get_data_chunk(train_df[train_df['approval_label'] == 'approved'], example_pool_size, random_state=random.randint(1, 10000000))
                rejected_data_chunk = get_data_chunk(train_df[train_df['approval_label'] == 'rejected'], example_pool_size, random_state=random.randint(1, 10000000))
            refinement_inference_data_chunk = pd.concat([approved_data_chunk, rejected_data_chunk], ignore_index=True)
            
            # 2 - evaluate on training examples
            df_refinement_inference_merged = evaluate_single_policy_domain(policy_domain_idx+1, policy_domain, best_rules, 
                                                                           refinement_inference_data_chunk, prediction_model, exp_name=f'Eval on train, iter {iter_id}')
            
            eval_report_train_before, _ = calculate_eval_metric(df_refinement_inference_merged, is_print=False)
            f1_train_before = eval_report_train_before.get('macro avg').get('f1-score')
            
            print('F1 score iter {} (previous best, before revision): train {:.04f}, valid {:.04f}'.format(iter_id, f1_train_before, f1_valid_best))
            
            # 3 - generate revised prompt
            refinement_system_prompt = Template(refinement_prompt).render(policy_domains=[(f'{policy_domain_idx+1}. {policy_domain}', pd.DataFrame.from_dict(best_rules))])

            df_refinement_inference_merged.rename(
                columns={
                    'approval_label':'true_approval_label', 
                    'compliance_status':'your_colleague_approval_label', 
                    'comment': 'your_colleague_comment'
                }, inplace=True)
            # separate the df_refinement_inference_merged into tp, fp, tn, fn subsets for user prompt
            df_tp = df_refinement_inference_merged[(df_refinement_inference_merged.true_approval_label == 'rejected') 
                                                   & (df_refinement_inference_merged.your_colleague_approval_label == 'REJECTED')]
            df_tn = df_refinement_inference_merged[(df_refinement_inference_merged.true_approval_label == 'approved') 
                                                   & (df_refinement_inference_merged.your_colleague_approval_label == 'APPROVED')]
            df_fp = df_refinement_inference_merged[(df_refinement_inference_merged.true_approval_label == 'approved') 
                                                   & (df_refinement_inference_merged.your_colleague_approval_label == 'REJECTED')]
            df_fn = df_refinement_inference_merged[(df_refinement_inference_merged.true_approval_label == 'rejected') 
                                                   & (df_refinement_inference_merged.your_colleague_approval_label == 'APPROVED')]
            refinement_label_to_data_chunk_dict = {
                'True Positives (Correctly rejected)': df_tp,
                'True Negatives (Correctly approved)': df_tn,
                'False Positives (Incorrectly rejected)': df_fp,
                'False Negatives (Incorrectly approved)': df_fn,
            }
            
            if example_ranking_strategy:
                if example_ranking_strategy == 'balanced':
                    assert 'n_examples' in example_ranking_strategy_args
                    df_tp = df_tp[:example_ranking_strategy_args['n_examples']]
                    df_tn = df_tn[:example_ranking_strategy_args['n_examples']]
                    df_fp = df_fp[:example_ranking_strategy_args['n_examples']]
                    df_fn = df_fn[:example_ranking_strategy_args['n_examples']]
                else:
                    raise NotImplementedError
            
            print({k: len(v) for k, v in refinement_label_to_data_chunk_dict.items()})
        
            if sum([df.shape[0] for df in refinement_label_to_data_chunk_dict.values()]) != df_refinement_inference_merged.shape[0]:
                print(f'Warning: refinement data size mismatched for policy domain: {policy_domain}, iteration: {iter_id}')
                print(f'Total size of refinement data: {sum([df.shape[0] for df in refinement_label_to_data_chunk_dict.values()])}, expected size: {df_refinement_inference_merged.shape[0]}')
            refinement_data_user_prompt = convert_dataset_to_user_prompt(refinement_label_to_data_chunk_dict, 
                                                                         components_override=refinement_required_fields, included_comment_col='your_colleague_comment')
            refinement_data_llm_request = build_llm_gateway_request(system_prompt=refinement_system_prompt, user_prompt=refinement_data_user_prompt, model=rules_generation_model,
                                                                    max_tokens=4096, temperature=0.5, seed=seed)
            try:
                refinement_responses = call_llm_gateway('uclanlp', refinement_data_llm_request)
                cur_iter_proposal = json.loads(refinement_responses.choices[0].message.content.strip('`json\n'))
            except:
                print('Exception during refinement. Using the previous round\'s rule as the prediction')
                cur_iter_proposal = cur_policy_metrics[-1]['cur_iter_proposal']

            cur_iter_proposed_rules = cur_iter_proposal['policy_rules']
            print(json.dumps(cur_iter_proposed_rules, indent=4))
            
            # 4 - evaluate the revised rule
            eval_report_valid_after, _ = calculate_eval_metric(evaluate_single_policy_domain(policy_domain_idx+1, policy_domain, cur_iter_proposed_rules, 
                                                               valid_data_chunk, prediction_model, exp_name=f'Eval on valid, iter {iter_id}'), is_print=False)
            f1_valid_after = eval_report_valid_after.get('macro avg').get('f1-score')
            
            eval_report_train_after, _ = calculate_eval_metric(evaluate_single_policy_domain(policy_domain_idx+1, policy_domain, cur_iter_proposed_rules, 
                                                               refinement_inference_data_chunk, prediction_model, exp_name=f'Eval on train, iter {iter_id}'), is_print=False)
            f1_train_after = eval_report_train_after.get('macro avg').get('f1-score')
            
            print('F1 score iter {} (after revision): train {:.04f}, valid {:.04f}'.format(iter_id, f1_train_after, f1_valid_after))
            
            # 5 - update and save metrics
            if improvement_criterion == 'train':
                improve_flag =  f1_train_before <= f1_train_after
            else:
                improve_flag = f1_valid_best < f1_valid_after and f1_train_before <= f1_train_after
            if improve_flag:
                if f1_valid_best < f1_valid_after:
                    print('Validation metric improved: {} > {}'.format(f1_valid_after, f1_valid_best))
                else:
                    print('Train metric improved: {} > {}'.format(f1_train_after, f1_train_before))
                best_rules = cur_iter_proposed_rules
                proposal_best_iter = cur_iter_proposal
                best_iter = iter_id
                eval_report_valid_before = valid_report_best_iter = eval_report_valid_after
                f1_valid_best = f1_valid_after
            else:
                if f1_train_before <= f1_train_after:
                    print('Train metric does not improve')
                else:
                    print('Validation metric does not improve: {} <= {}'.format(f1_valid_after, f1_valid_best))
            
            cur_policy_metrics.append({
                'iter': iter_id,
                'cur_iter_proposal': cur_iter_proposal,
                'eval_report_valid_prev_best_iter': eval_report_valid_before,
                'eval_report_valid_after': eval_report_valid_after,
                'eval_report_train_prev_best_iter': eval_report_train_before,
                'eval_report_train_after': eval_report_train_after
            })
            
            # save generated policy
            json.dump(cur_policy_metrics[-1], open(cur_round_output_dir + f'/round_metrics_{policy_domain}.json', 'w'), indent=4)
            
        # log the best prompt
        final_log = {
            'best_iter': best_iter,
            'best_iter_proposal': proposal_best_iter,
            'eval_report_valid_best_iter': valid_report_best_iter,
        }
        print(json.dumps(final_log))
        
if __name__ == '__main__':
    policy_domains = ['Copyrights and Competitive Claims', 'Endorsement', 'Exploitative', 'Finance Claims', 'Health Claims', 
                      'Misrepresentative', 'Offensive', 'Politicized', 'Quality', 'Sexualized or Skin']
    
    rules_generation_model = 'gpt-4-turbo-2024-04-09' # 'gpt-4-turbo-2024-04-09-research'
    prediction_model = "gpt-4o-2024-08-06-research"
    
    seed = 2024
    in_dir = f'/home/diwu/research_with_Taboola/abby-taboola/meta_prompting/results/single_iter_meta_prompt/30approved_70rejected_seed{seed}/rules/'
    main(in_dir, rules_generation_model, prediction_model, working_policy_domains=['Copyrights and Competitive Claims'], seed=seed,
         example_pool_size=60, example_ranking_strategy='balanced', example_ranking_strategy_args={'n_examples': 10})