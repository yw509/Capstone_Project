import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import openai
from openai import OpenAI, AzureOpenAI
import backoff
import os
import json

taboola_client = AzureOpenAI(
    azure_endpoint = "https://qa-la-002.openai.azure.com/",
    api_key=open('api_key_taboola.txt').readline().strip(),
    api_version="2024-02-15-preview"
)
rules_generation_model = "gpt-4-turbo-2024-04-09-research"  # 'dev-openai-gpt4-vision-preview' # dev-openai-o1-preview       # Model for rules generation
image_check_model = "gpt-4o-2024-08-06-research"  # 'dev-azure-gpt4o'     # Model for image availability checking
evaluation_model = "gpt-4o-2024-08-06-research"  # Model for evaluation

uclanlp_client = OpenAI(
    api_key=open('api_key_taboola_openai.txt').readline().strip(),
)
rules_generation_model = 'gpt-4-turbo-2024-04-09'

@backoff.on_exception(backoff.constant, (openai.RateLimitError, openai.APIError), interval=2)
def openai_chat_completion_with_backoff(backend, **kwargs):
    if backend == 'uclanlp':
        response = uclanlp_client.chat.completions.create(**kwargs)
    else:
        response = taboola_client.chat.completions.create(**kwargs)
    return response

evaluation_prompt = """
**Context:**
You are a senior content reviewer with twenty years experiences and tasked with evaluating advertisements for policy compliance based on a set of rules.

**Task:**
You will be given policy rules which are combined with policy domain(s) and a set of policy rules for each domain.
You will also be provided with a dataset of ads for evaluation.
Your task is to evaluate the ads based on the rules and determine whether they are compliant or non-compliant.
When evaluating the ads, please make sure to:
  - Review all components of the ad, including the title, description, thumbnail, etc.
  - For each ad, provide a clear explanation of why it is compliant or non-compliant based on the rules.
    - If the ad violates a rule/rules, list the rule id(s) it violates.
  - Review all the ads in the dataset.

**Policy Rules:**
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
  "ad_evaluations": [
    {
      "ad_id": "Unique identifier for the ad",
      "comment": "Explanation of why the ad is compliant(approved) or non-compliant(rejected).",
      "violated_rules": ["1.1", "3.2", ...],  // List of rule IDs violated by the ad
      "compliance_status": "APPROVED" or "REJECTED",  // "REJECTED" if any violated rules, otherwise "APPROVED"
    },
  ]
}
"""

def get_data_chunk(data, chunk_size, random_state):
    """
    Get a chunk of data from the dataset.

    Parameters:
        data (pd.DataFrame): The dataset to sample from.
        chunk_size (int): The size of the chunk to sample.
        random_state (int): The random seed for reproducibility.

    Returns:
        pd.DataFrame: The sampled data chunk. If the chunk size is larger than the dataset, the entire dataset is returned.
    """
    if chunk_size >= data.shape[0]:
        data_chunk = data
    else:
        data_chunk = data.sample(n=chunk_size, random_state=random_state)
    return data_chunk

# updated version with public Azure API
def get_data_chunk_with_valid_thumbnail_url(data_chunk, img_detail_level):
    """
    Get the valid data from the data chunk by checking thumbnail URLs.

    Parameters:
        data_chunk (pd.DataFrame): The data chunk to filter.

    Returns:
        pd.DataFrame: The valid data from the data chunk.
    """
    thumbnail_urls = data_chunk.thumbnail_url.to_list()
    valid_responses = []

    # Loop through each URL and create an Azure OpenAI request for it
    for url in tqdm(thumbnail_urls, desc='URL thumbnail check'):
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps verify if a thumbnail is inappropriate."},
            {"role": "user", "content": [
                {"type": "text", "text": f'\nThumbnail: '},
                {
                  "type": "image_url",
                  "image_url": {
                      "url": url,
                      "detail": img_detail_level
                  },
                },
            ]}
        ]
        
        # Call the Azure OpenAI API
        try:
            response = openai_chat_completion_with_backoff(
                backend='taboola',
                model=image_check_model,
                messages=messages,
                temperature=0.0,
                max_tokens=20,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
            )
        except:
            valid_responses.append(False)
            continue

        # Check the response and determine if it's valid
        if 'inappropriate' not in response.choices[0].message.content.lower():
            valid_responses.append(True)
        else:
            valid_responses.append(False)

    # Filter the data chunk based on valid responses
    data_chunk = data_chunk[valid_responses]
    return data_chunk, valid_responses

def convert_dataset_to_user_prompt(label_to_data_chunk_dict, components_override=None, is_relevant_component_only=False, included_comment_col=None, img_detail_level='low'):
    if components_override is not None and is_relevant_component_only:
        raise ValueError('Cannot use both components_override and is_relevant_component_only')
    
    default_components = ['title', 'description', 'thumbnail_url']
    components = components_override if components_override is not None else default_components

    # Initialize the content list
    content = []
    
    for label, data_chunk in label_to_data_chunk_dict.items():
        if data_chunk is not None:
            content.append({"type": "text", "text": f'{label}:'})
            
            for index, row in data_chunk.iterrows():
                content.append({"type": "text", "text": f'{index + 1}.'})
                
                required_components = row['relevant_component'] if is_relevant_component_only else components
                for component in required_components:
                    if component == 'thumbnail_url':
                        # Using the specified format for image URLs
                        content.append({"type": "text", "text": '\nThumbnail:'})
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": row[component],
                                "detail": img_detail_level
                            }
                        })
                    else:
                        content.append({"type": "text", "text": f'{component}: {row[component]}'})

                # If there's an included comment column, add it
                if included_comment_col is not None and row[included_comment_col]:
                    content.append({"type": "text", "text": f'review comment: {row[included_comment_col]}'})

                content.append({"type": "text", "text": ''})  # Add a blank text entry for spacing
    
    # Return a single user message structure
    return {
        "role": "user",
        "content": content
    }

def build_llm_gateway_request(system_prompt, user_prompt, model, max_tokens=1000, temperature=0.5, seed=None):
    """
    Build the request payload for the OpenAI API and call the completion endpoint with backoff.

    Parameters:
        system_prompt (str): The system-level instruction for the model.
        user_prompt (list): A list of user messages and other prompts in the required format.
        model (str): The model deployment to use for the request.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The sampling temperature for the model.
        seed (int or None): Random seed for the model (optional, not used in OpenAI API).

    Returns:
        dict: The response from the OpenAI API.
    """
    # Construct the messages payload
    messages = [
        {
            "role": "system",
            "content": system_prompt  # System-level prompt
        },
        user_prompt
    ]

    # Prepare the arguments for the OpenAI chat completion
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    return kwargs

def call_llm_gateway(backend, kwargs):
    return openai_chat_completion_with_backoff(backend=backend, **kwargs)

def call_llm_gateway_concurrently(llm_gateway_req_list, max_concurrency=3):
    call_llm_gateway_partial = lambda x: call_llm_gateway('taboola', x)
    # Use ThreadPoolExecutor to handle multiple requests in parallel
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        responses = list(tqdm(executor.map(call_llm_gateway_partial, llm_gateway_req_list), total=len(llm_gateway_req_list)))
    return responses

def calculate_eval_metric(df_eval, is_print=False):
    y_pred = df_eval['compliance_status'].map({'APPROVED': 'approved', 'REJECTED': 'rejected'})
    y_true = df_eval['approval_label']
    report= classification_report(y_true, y_pred, labels=['approved', 'rejected'], output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=['approved', 'rejected'])
    cm_df = pd.DataFrame(cm, index=['Actual Approved', 'Actual Rejected'], columns=['Predicted Approved', 'Predicted Rejected'])
    if is_print:
        print(report)
        print(cm_df)
    return report, cm_df

def get_unique_top_n(bm25_model, eval_content, data, similar_item, remove_duplicate_titles):
    top_items = bm25_model.get_top_n(eval_content, data.to_dict('records'), n=len(data))
    
    if remove_duplicate_titles:
        title_counts = {}
        for ad in top_items:
            title_counts[ad['title']] = title_counts.get(ad['title'], 0) + 1
        
        unique_top_items = []
        for ad in top_items:
            if title_counts[ad['title']] == 1: 
                unique_top_items.append(ad)
                if len(unique_top_items) == similar_item:  
                    break
        
        return unique_top_items
    else:
        return top_items[:similar_item]

def get_unique_top_random(data, similar_item, remove_duplicate_titles):

    if remove_duplicate_titles:
        title_counts = data['title'].value_counts()
        unique_data = data[data['title'].map(title_counts) == 1] 
        if len(unique_data) >= similar_item:
            return unique_data.sample(n=similar_item)
        else:
            return unique_data
    else:
        if len(data) >= similar_item:
            return data.sample(n=similar_item)
        else:
            return data

def get_all_policy_rules():
    # Directory containing JSON files
    directory = "/Users/wangyuchen/Desktop/research_with_Taboola/abby-taboola/meta_prompting/single_iter_meta_prompt/rules_for_in_context_learning"

    # Dictionary to hold policy domain names and their corresponding policy rules
    policy_rules_dict = {}

    # Iterate over each JSON file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            
            # Open and load the JSON data
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Extract the policy domain name and policy rules
                policy_domain = data["cur_iter_proposal"]["policy_domain"]
                policy_rules = data["cur_iter_proposal"]["policy_rules"]
                
                # Add the policy rules to the dictionary under the policy domain name
                policy_rules_dict[policy_domain] = policy_rules

    return policy_rules_dict
