import json
import argparse
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import os



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

    def generate_responses(self, inputs,meta_data) -> List[Dict[str, Any]]:
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

            responses.append([{"role": "user", "content": content["prompt"]}])
        
        # Prepare the inputs for the LLM generation
        sampling_params = SamplingParams(temperature=0, max_tokens=4096)
        resp_inputs = [
            tokenizer.apply_chat_template(inst, tokenize=False, add_generation_prompt=True) 
            for inst in responses
        ]
        

        resp_outputs = self.llm_engine.generate(resp_inputs, sampling_params)


        return [
            {   
                "meta_data":meta,
                "input": resp_input, 
                "output": resp_output.outputs[0].text.strip().lstrip("\n").rstrip("\n"),
                "golden": input["answers"]
            }
            for resp_input, resp_output, input, meta in zip(resp_inputs, resp_outputs,inputs,meta_data)
        ]

def data_loader(file_path, debug=False):
    # 读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 打印所有内容
    input_data = []
    meta_data = []
    for i, entry in enumerate(tqdm(data)):
        # import pdb
        # pdb.set_trace()
        
        if "fever" in file_path:
            prompt_format = f"""Give the following passages, try to answer this question with step-by-step thoughts and make the answer more structural. Please generate SUPPORTS or REFUTES as the final answer directly after ####:
            Passage: {entry['retrieval_docs'][0]["contents"]}
            Passage: {entry['retrieval_docs'][1]["contents"]}
            Passage: {entry['retrieval_docs'][2]["contents"]}
            Passage: {entry['retrieval_docs'][3]["contents"]}
            Passage: {entry['retrieval_docs'][4]["contents"]}
            Give the following passages, try to answer this question with step-by-step thoughts and make the answer more structural. Please generate SUPPORTS or REFUTES as the final answer directly after ####:
            Question: {entry['question']}"""

        elif "wikiasp" in file_path:

            prompt_format = f"""Give the following passages, Sub-aspects: {entry['aspect']}. please generate a summary about {entry['question']} directly after ####:
            Passage: {entry['retrieval_docs'][0]["contents"]}
            Passage: {entry['retrieval_docs'][1]["contents"]}
            Passage: {entry['retrieval_docs'][2]["contents"]}
            Passage: {entry['retrieval_docs'][3]["contents"]}
            Passage: {entry['retrieval_docs'][4]["contents"]}
            Give the following passages, Sub-aspects: {entry['aspect']}. please generate a summary about {entry['question']} directly after ####"""

        else:

            prompt_format = f"""Give the following passages, try to answer this question with step-by-step thoughts and make the answer more structural. Please generate the final answer directly after ####:
            Passage: {entry['retrieval_docs'][0]["contents"]}
            Passage: {entry['retrieval_docs'][1]["contents"]}
            Passage: {entry['retrieval_docs'][2]["contents"]}
            Passage: {entry['retrieval_docs'][3]["contents"]}
            Passage: {entry['retrieval_docs'][4]["contents"]}
            Give the following passages, try to answer this question with step-by-step thoughts and make the answer more structural. Please generate the final answer directly after ####
            Question: {entry['question']}"""

        # import pdb
        # pdb.set_trace()
        
        data_format = {
            "answers": entry['golden_answers'],
            "prompt": prompt_format
        }

        input_data.append(data_format)
        meta_data.append(entry)

        # print(data_format)
    
    if debug:
        input_data = input_data[:10]
        print("Debug mode enabled. Processing a smaller subset of data.")


    return input_data, meta_data



if __name__ == "__main__":
    # Argument parsing for command-line execution
    parser = argparse.ArgumentParser(description="Run Response Generation")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--inst_file', type=str, required=True, help='Path to the input JSON file with instructions')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the input JSON file with instructions')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (process a smaller subset)')
    parser.add_argument('--dataset_name', type=str, required=True, help='name for output')

    args = parser.parse_args()
    


    input_data, meta_data = data_loader(args.inst_file,args.debug)
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

    output_path = args.output_path + f'responses_{model_name}_{args.dataset_name}.json'

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save responses to a JSON file
    with open(output_path, 'w') as file:
        for response in responses:
            json.dump(response, file)
            file.write('\n')


    