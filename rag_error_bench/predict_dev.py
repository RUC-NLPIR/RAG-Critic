import json
import argparse
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import os
# import jsonlines
import json_repair


class ResponseGenerator:
    def __init__(self, model_name_or_path: str) -> None:
        """
        Initializes the response generator with the model path.

        Args:
            model_name_or_path (str): The path to the model or the model name.
        """
        self.model_name_or_path = model_name_or_path
        self.llm_engine = self.create_llm_engine(model_name_or_path)

    def create_llm_engine(self, model_name_or_path: str) -> LLM:
        """
        Creates an LLM engine for generating responses.

        Args:
            model_name_or_path (str): The model path or model name.

        Returns:
            LLM: An instance of the LLM engine.
        """
        return LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            tokenizer_mode="auto",
            tensor_parallel_size=4
            # dtype=torch.bfloat16
            # dtype="bfloat16"
        )
# gpu_memory_utilization=0.95
    def generate_responses(self, inputs, meta_data) -> List[Dict[str, Any]]:
        """
        Generate responses for the provided instructions and inputs.

        Args:
            instructions (List[str]): List of instructions to be used for response generation.
            inputs (List[str]): List of inputs corresponding to the instructions.
            outputs (List[str]): List of outputs for evaluation (currently not used in the response generation).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with instructions, inputs, and the generated responses.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        responses = []
        outputs=[]
 
        for content in inputs:
            # import pdb
            # pdb.set_trace()

            # responses.append([{"role": "user", "content": content}])
            responses.append([{"role": "system", "content": content},{"role": "user", "content": ""}])
        
        # Prepare the inputs for the LLM generation
        sampling_params = SamplingParams(temperature=0, max_tokens=4096)
        resp_inputs = [
            tokenizer.apply_chat_template(inst, tokenize=False, add_generation_prompt=True) 
            for inst in responses
        ]
        

        resp_outputs = self.llm_engine.generate(resp_inputs, sampling_params)

        # import pdb
        # pdb.set_trace()

        final_answers = []
        for resp_input, resp_output,entry in zip(resp_inputs, resp_outputs,meta_data):
            try:
                # import pdb
                # pdb.set_trace()
                final_answers.append({
                    "input": entry['instruction'],
                    "golden":entry['output'],
                    "output":  json_repair.loads(resp_output.outputs[0].text.strip().lstrip("\n").rstrip("\n"))
                })
            except:
                continue
        return final_answers




# # json_repair.loads()[]

#         return [
#             {
#                 "input": resp_input, 
#                 "output":  json_repair.loads(resp_output.outputs[0].text.strip().lstrip("\n").rstrip("\n"))['Error_analysis']
#             }
#             for resp_input, resp_output in zip(resp_inputs, resp_outputs)
#         ]

def data_loader(file_path, debug=False):


    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 忽略空行
                data.append(json.loads(line))

  

    # 打印所有内容
    input_data = []
    meta_data = []
    for i,entry in enumerate(tqdm(data)):

        

        # prompt_format = f"""You are an expert in error analysis for retrieval-augmented generation tasks. We will provide you with a prompt that includes both the question and relevant knowledge, along with a model's prediction and the golden answer. The details are as follows:
        # prompt: {entry['input']}
        # model's prediction: {entry['output']}
        # golden answer: {entry['golden']}
        # If the model's prediction is incorrect, please respond with a single JSON including the judgement in key 'Judgement' and a detailed error analysis in key 'Error_analysis'.
        # Here is an example of output JSON format: {{'Judgement': "incorrect", 'Error_analysis': "The model's prediction is incorrect because ..."}}
        # If the model's prediction is correct, please respond with a single JSON as follows:
        # {{'Judgement': "correct", 'Error_analysis': "None"}}
        # """
        prompt_format = entry['instruction']
        
        
        # prompt_format = f"""You are an expert in error analysis for retrieval-augmented generation tasks. We will provide you with a prompt that includes both the question and relevant knowledge, along with a model's prediction and the golden answer. The details are as follows:
        # prompt: {entry['input']}
        # model's prediction: {entry['output']}
        # golden answer: {entry['golden']}
        
        # Possible error types include:
        # 1. Noise Information: Irrelevant documents lead to ineffective answers.
        # 2. Information Accuracy: Documents contain factual errors or outdated information.
        # 3. Information Completeness: Documents lack key information for a complete answer.
        # 4. Insufficient Relevance: Low relevance of documents affects answer accuracy.
        # 5. Context Loss: Failure to utilize key information from documents during generation.
        # 6. Generation of Misleading Content: Generated answers do not reflect actual facts.
        # 7. Information Integration Failure: Inability to integrate information from multiple documents.
        # 8. Omission of Important Information: Key factual information is omitted in the answer.
        
        # If the model's prediction is incorrect, please respond with a single JSON including the error type in key 'Error_type' and a detailed error analysis in key 'Error_analysis'.
        # Here is an example of output JSON format: {{'Error_type': str, 'Error_analysis': str}}
        # If the model's prediction is correct, please respond with a single JSON as follows:
        # {{'Error_type': correct, 'Error_analysis': None}}
        # """

        input_data.append(prompt_format)
        meta_data.append(entry)

        # print(data_format)
    
    if debug:
        input_data = input_data[:50]
        print("Debug mode enabled. Processing a smaller subset of data.")


    return input_data,meta_data



if __name__ == "__main__":
    # Argument parsing for command-line execution
    parser = argparse.ArgumentParser(description="Run Response Generation")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--inst_file', type=str, required=True, help='Path to the input JSON file with instructions')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the input JSON file with instructions')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (process a smaller subset)')
    parser.add_argument('--dataset_name', type=str, required=True, help='name for output')

    args = parser.parse_args()
    


    input_data,meta_data = data_loader(args.inst_file,args.debug)
    print(input_data[0])

    response_generator = ResponseGenerator(args.model_name_or_path)

    # Generate responses using the loaded instructions and inputs
    responses = response_generator.generate_responses(
        input_data, meta_data
    )
    # import pdb
    # pdb.set_trace()
    model_name = args.model_name_or_path.split("/")[-1]
    file_name = args.inst_file.split("/")[-1].split(".")[0]
    # import pdb
    # pdb.set_trace()

    # output_path = f'/share/project/dgt/vllm_sample_jjj/output/critic_{model_name}_{args.dataset_name}.json'

    output_path = args.output_path + f'predict_{model_name}_{args.dataset_name}.json'
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save responses to a JSON file
    with open(output_path, 'w') as file:
        for response in responses:
            json.dump(response, file)
            file.write('\n')


    