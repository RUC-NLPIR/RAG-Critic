import os
import json
import argparse
import json_repair
from prompt_template import SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE
from flashrag.config import Config
from flashrag.utils import get_dataset, get_generator
from flashrag.prompt import PromptTemplate
from flashrag.evaluator import Evaluator

# Function to run the evaluation and generation process
def run(
    dataset_name,
    split,
    model_name,
    plan_model_name,
    plan_model_path,
    gpu_id,
    test_sample_num,
    save_dir,
    save_note,
    retrieval_data_dir,
    previous_answer_data_dir,
    error_data_dir,
    config_path,
):
    """
    Run the evaluation process for a given dataset, model, and configuration.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'nq', 'triviaqa').
        split (str): The data split ('test' or 'dev').
        model_name (str): The model name used for answering.
        plan_model_name (str): The model name used for generation.
        generator: The generator object used to generate predictions.
        test_sample_num (int): The number of test samples to process.
        save_note (str): A note for saving results.
    """

    config_dict = {
        "save_dir": save_dir,
        "dataset_name": dataset_name,
        "split": ['dev','test'],
        "framework": "vllm",
        "generator_model": plan_model_name,
        "generator_model_path": plan_model_path,
        "gpu_id": gpu_id,
        "test_sample_num": test_sample_num,
        "save_note": save_note
    }

    # Initialize config and generator
    config = Config(config_path, config_dict)
    save_path = os.path.join(save_dir, f'{dataset_name}_{config_dict["generator_model"]}_{model_name}_{config_dict["test_sample_num"]}_{save_note}.json')

    # Load generator
    generator = get_generator(config)
    
    # Load dataset and prompt templates
    dataset = get_dataset(config)[split]
    prompt_template = PromptTemplate(config, system_prompt=SYSTEM_PROMPT_TEMPLATE, user_prompt=USER_PROMPT_TEMPLATE)

    # Define paths for input data
    retrieval_data_path = os.path.join(retrieval_data_dir, f'{dataset_name}/{split}.json')
    previous_answer_data_path = os.path.join(
        previous_answer_data_dir,
        f'responses_{model_name}_{dataset_name}_{split}_100.json'
    )
    error_data_path = os.path.join(error_data_dir, f'errordata_{dataset_name}_{model_name}_{split}.json')


    # Read previous answers, retrieval data, and error data
    with open(previous_answer_data_path, 'r') as f:
        previous_answer_data = [json_repair.loads(line) for line in f]
    with open(retrieval_data_path, 'r') as f:
        retrieval_data = json.load(f)
    with open(error_data_path, 'r') as f:
        error_data = [json.loads(line) for line in f]

    # Mapping of questions to errors
    question2error = {}
    for item in error_data:
        input_str = item['input']
        question = input_str.split("\n\n")[1].split("\n")[0].replace("Question:", "").strip()
        try:
            question2error[question] = item['output']['tag2']
        except:
            continue

    # Generate prompts for the dataset
    prompt_list = []
    for idx, item in enumerate(dataset):
        question = item.question
        error_type = 'No Error' if question not in question2error else ",".join(question2error[question])

        previous_answer = previous_answer_data[idx]['output'].split("So the final answer is:")[-1].strip()
        retrieval_result = retrieval_data[idx]
        assert question == retrieval_result['question']
        doc_list = [d['contents'] for d in retrieval_result['retrieval_docs']]

        prompt = prompt_template.get_string(question=question, doc_list=str(doc_list), previous_pred=previous_answer, error_type=error_type)
        prompt_list.append(prompt)

    # Generate predictions using the generator
    output_list = generator.generate(prompt_list, max_tokens=2048, temperature=0.0)

    # Save the results to the dataset
    dataset.update_output('input_prompt', prompt_list)
    dataset.update_output('pred', output_list)
    dataset.save(save_path)

# Main function for argument parsing and execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help="Name of the model to use for predictions.")
    parser.add_argument('--plan_model_name', type=str, default='qwen2.5-72B-instruct')
    parser.add_argument('--plan_model_path', type=str, default= 'Qwen/Qwen2.5-72B-Instruct')
    parser.add_argument('--sample_num', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='results/plan/')
    parser.add_argument('--save_note', type=str, help="Note for saving results.")
    parser.add_argument('--dataset_name', type=str, default='nq')
    parser.add_argument('--split', type='str', default='test')
    parser.add_argument('--gpu_id', type='str', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--retrieval_data_dir', type='str', default='retrieval_results/')
    parser.add_argument('--previous_answer_data_dir', type='str', default='previous_answer_data/')
    parser.add_argument('--error_data_dir', type='str', default='error_data/')
    parser.add_argument('--config_path', type=str, default='myconfig.yaml')
    args = parser.parse_args()

    run(
        dataset_name=args.dataset_name, 
        split=args.split, 
        model_name=args.model_name, 
        plan_model_name=args.plan_model_name, 
        plan_model_path=args.plan_model_path, 
        gpu_id=args.gpu_id, 
        test_sample_num=args.sample_num, 
        save_dir=args.save_dir, 
        save_note=args.save_note,
        retrieval_data_dir=args.retrieval_data_dir,
        previous_answer_data_dir=args.previous_answer_data_dir,
        error_data_dir=args.error_data_dir,
        config_path=args.config_path
    )
