import os
import json
import argparse
import warnings
import json_repair
from typing import List

# Set environment variable for VLLM
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from agent_prompt_template import *
from flashrag.utils import get_generator, get_retriever, get_dataset
from flashrag.config import Config
from flashrag.evaluator import Evaluator

class CriticAgent:
    """
    An agent that performs RAG-based question answering with self-criticism and correction capabilities.
    
    This agent can:
    1. Retrieve relevant documents
    2. Rewrite and decompose queries
    3. Refine documents
    4. Generate answers with self-correction
    """
    
    def __init__(self, generator, tokenizer, retriever):
        """
        Initialize the CriticAgent.
        
        Args:
            generator: Model for text generation
            tokenizer: Tokenizer for the generator model
            retriever: Document retriever
        """
        self.generator = generator
        self.retriever = retriever
        self.tokenizer = tokenizer
    
    def _fetch_final_answer_var_name(self, code_snippet: str) -> str:
        """Extract the variable name containing the final answer from code."""
        code_lines = code_snippet.split("\n")
        var_name = 'answer'
        for line in code_lines[::-1]:
            if line.startswith("#") or 'print' in line:
                continue
            if 'GenerateAnswer' in line:
                var_name = line.split('=')[0].strip()
                break
        return var_name
    
    def _get_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Create a formatted prompt for the model."""
        input_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        input = self.tokenizer.apply_chat_template(input_list, tokenize=False, add_generation_prompt=True)
        return input

    def _parse_code_snippet(self, agent_output: str) -> str:
        """Parse and process the code snippet from agent output."""
        func_map = {
            'Retrieval': "self._Retrieval",
            'RewriteQuery': "self._RewriteQuery",
            'DecomposeQuery': "self._DecomposeQuery",
            'RefineDoc': "self._RefineDoc",
            'GenerateAnswer': "self._GenerateAnswer"
        }
        code_snippet = agent_output.split('```python')[-1].split('```')[0].strip()
        for r_func, t_func in func_map.items():
            if r_func in code_snippet:
                code_snippet = code_snippet.replace(r_func, t_func)
        print("----code snippet after parse----")
        print(code_snippet)
        return code_snippet

    def run(self, question: str, doc_list: List[str], previous_answer: str, agent_output: str) -> tuple[bool, str]:
        """
        Execute the agent's reasoning process.
        
        Args:
            question: The input question
            doc_list: List of relevant documents
            previous_answer: Previous model's answer
            agent_output: Output from the planning agent
            
        Returns:
            tuple: (success_flag, new_answer)
        """
        code_snippet = self._parse_code_snippet(agent_output)
        
        global_namespace = globals()
        global_namespace.update({
            'question': question,
            'doc_list': doc_list,
            'previous_pred': previous_answer,
            'self': self,
        })
        
        try:
            exec(code_snippet, global_namespace)
            final_answer_var_name = self._fetch_final_answer_var_name(code_snippet)
            new_answer = global_namespace[final_answer_var_name]
            return True, new_answer
        except:
            print("have an error!")
            return False, ""
    
    def _Retrieval(self, query: str, topk: int) -> List[str]:
        """Retrieve relevant documents for a query."""
        doc_list = self.retriever.search(query, topk)
        return [item['contents'] for item in doc_list]
    
    def _RewriteQuery(self, query: str, instruction: str) -> List[str]:
        """Rewrite the query based on instruction."""
        if instruction == 'clarify':
            system_prompt = REWRITE_CLARIFY_QUERY_PROMPT['system_prompt']
            user_prompt = REWRITE_CLARIFY_QUERY_PROMPT['user_prompt'].format(query=query)
        elif instruction == 'expand':
            system_prompt = REWRITE_EXPAND_QUERY_PROMPT['system_prompt']
            user_prompt = REWRITE_EXPAND_QUERY_PROMPT['user_prompt'].format(query=query)
        else:
            # user specific instruction
            system_prompt = REWRITE_CUSTOM_QUERY_PROMPT['system_prompt'].format(instruction=instruction)
            user_prompt = REWRITE_CUSTOM_QUERY_PROMPT['user_prompt'].format(query=query)
        prompt = self._get_prompt(system_prompt, user_prompt)
        output = self.generator.generate(prompt, max_tokens=2048, temperature=0.0)[0]

        # parse output
        output = json_repair.loads(output)
        if isinstance(output, dict):
            output = output['query']
            if isinstance(output, str):
                output = [output.strip()]

        # check output 
        if not all([isinstance(q, str) for q in output]):
            warnings.warn("Output is not a list of strings")

        return output
    
    def _DecomposeQuery(self, query: str) -> List[str]:
        """Decompose a complex query into simpler sub-queries."""
        system_prompt = DECOMPOSE_QUERY_PROMPT['system_prompt']
        user_prompt = DECOMPOSE_QUERY_PROMPT['user_prompt'].format(query=query)
        prompt = self._get_prompt(system_prompt, user_prompt)
        output = self.generator.generate(prompt, max_tokens=2048, temperature=0.0)[0]

        # parse output
        output = json_repair.loads(output)
        # check output 
        if (not isinstance(output, list)) or (not all([isinstance(q, str) for q in output])):
            warnings.warn("Output format is wrong. Output is not a list of strings")

        return output

    def _RefineDoc(self, query: str, doc: str, instruction: str) -> str:
        """Refine a document based on the query and instruction."""
        if instruction == 'explain':
            system_prompt = REFINE_DOC_EXPLAIN_PROMPT['system_prompt']
            user_prompt = REFINE_DOC_EXPLAIN_PROMPT['user_prompt'].format(document=doc, question=query)
        elif instruction == 'summarize':
            system_prompt = REFINE_DOC_SUMMARIZE_PROMPT['system_prompt']
            user_prompt = REFINE_DOC_SUMMARIZE_PROMPT['user_prompt'].format(document=doc, question=query)
        else:
            return doc
        prompt = self._get_prompt(system_prompt, user_prompt)
        output = self.generator.generate(prompt, max_tokens=2048, temperature=0.0)[0]

        output = json_repair.loads(output)
        try:
            if 'refined_document' in output:
                return output['refined_document']
            elif 'explanation' in output:
                return output['explanation']
            else:
                return doc
        except:
            warnings.warn("Output format is wrong. Output is not a JSON object")
            return doc
    
    def _GenerateAnswer(self, query: str, docs: List[str], additional_instruction: str = None) -> str:
        """Generate a final answer based on the query and documents."""
        if additional_instruction is None:
            tips = ""
        else:
            tips = "Here are some tips you need to pay attention to during the generation process: {additional_instruction}"
        system_prompt = tips
        user_prompt = """Find the useful content from the provided documents, then answer the question. Answer the question directly. Your response should be very concise. Please provide use ’So the final answer is:’ as a prefix for the final answer. The following are given documents. \n{reference}\nAnswer the question directly. Your response should be very concise. Please provide use ’So the final answer is:’ as a prefix for the final answer. **Question**: {question}**Response**:"""
        reference = ''
        for doc in docs:
            reference += f'Passage: {doc}\n'
        user_prompt = user_prompt.format(reference=reference, tips=tips, question=query)
        prompt = self._get_prompt(system_prompt, user_prompt)
        output = self.generator.generate(prompt, max_tokens=4096, temperature=0.0)[0]
        return output


def run(
    dataset_name: str,
    split: str,
    model_name: str,
    plan_model_name: str,
    execute_model: str,
    execute_model_path: str,
    test_sample_num: int,
    save_note: str,
    save_dir: str,
    plan_data_dir: str,
    retrieval_data_dir: str,
    previous_answer_data_dir: str,
    error_data_dir: str,
    gpu_id: str,
    retrieval_method: str,
    retrieval_model_path: str,
    config_path: str,
):
    """
    Run the execution agent on a dataset.
    
    Args:
        dataset_name: Name of the dataset
        split: Data split (test/dev)
        model_name: Name of the base model
        plan_model_name: Name of the planning model
        execute_model: Name of the execution model
        generator: Text generator object
        test_sample_num: Number of test samples
        save_note: Note for saving results
        save_dir: Directory for saving results
        retrieval_data_dir: Directory containing retrieval results
        previous_answer_data_dir: Directory containing previous model answers
        error_data_dir: Directory containing error analysis results
    """
    # Initialize configuration
    config_dict = {
        "save_note": f"{dataset_name}_{plan_model_name}_{execute_model}_{test_sample_num}_{save_note}",
        "save_dir": save_dir,
        "framework": "vllm",
        "generator_model": execute_model,
        "generator_model_path": execute_model_path,
        'retrieval_method': retrieval_method,
        'retrieval_model_path': retrieval_model_path,
        "gpu_id": gpu_id,
        'test_sample_num': test_sample_num
    }
    config = Config(config_path, config_dict)

    # Setup components
    generator = get_generator(config)
    retriever = get_retriever(config)
    tokenizer = generator.tokenizer
    agent = CriticAgent(generator, tokenizer, retriever)
    evaluator = Evaluator(config)
    dataset = get_dataset(config)[split]

    # Define paths
    plan_path = os.path.join(plan_data_dir, 
                            f'{dataset_name}_{plan_model_name}_{model_name}_{test_sample_num}_{save_note}.json')
    retrieval_data_path = os.path.join(retrieval_data_dir, f'{dataset_name}/{split}.json')
    previous_answer_data_path = os.path.join(
        previous_answer_data_dir,
        f'responses_{model_name}_{dataset_name}_{split}_100.json'
    )
    error_data_path = os.path.join(error_data_dir, f'errordata_{dataset_name}_{model_name}_{split}.json')

    # Load data
    with open(error_data_path, 'r') as f:
        error_data = [json.loads(line) for line in f]
    with open(previous_answer_data_path, 'r') as f:
        previous_answer_data = [json_repair.loads(line) for line in f]
    with open(retrieval_data_path, 'r') as f:
        retrieval_data = json.load(f)
    with open(plan_path, 'r') as f:
        agent_output_data = json.load(f)

    # Process each example
    for idx, item in enumerate(dataset):
        print(f"------------------processing {idx} / {len(dataset)}------------------")
        
        question = item.question
        previous_answer = previous_answer_data[idx]['output']
        previous_answer = previous_answer.split("So the final answer is:")[-1].strip()
        judgement = error_data[idx]['output']['Judgement']
        
        if 'error' in judgement.lower():
            retrieval_result = retrieval_data[idx]
            assert question == retrieval_result['question']
            doc_list = [d['contents'] for d in retrieval_result['retrieval_docs']]
            agent_output = agent_output_data[idx]['output']['pred']

            print("-"*100)
            print(f"question: {question}\nprevious_answer: {previous_answer}\nagent_output: {agent_output}")
            success_flag, new_answer = agent.run(question, doc_list, previous_answer, agent_output)
            print(f"new_answer: {new_answer}")
            
            new_answer_short = new_answer.replace('So the final answer is:','').strip()
            item.update_output('raw_pred', previous_answer)
            item.update_output('raw_doc_list', doc_list)
            item.update_output('agent_output', agent_output)
            item.update_output('success_flag', success_flag)
            item.update_output('new_pred', new_answer)
            item.update_output('pred', new_answer_short)
        else:
            print("original prediction is true!")
            item.update_output('success_flag', True)
            item.update_output('pred', previous_answer)
    
    # Evaluate results
    eval_result = evaluator.evaluate(dataset)
    print(eval_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--plan_model_name', type=str, default='qwen2.5-72B-instruct')
    parser.add_argument('--gpu_id', type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument('--sample_num', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='results/execute/')
    parser.add_argument('--plan_data_dir', type=str, default='results/plan/')
    parser.add_argument('--retrieval_data_dir', type=str, default='retrieve_results/')
    parser.add_argument('--previous_answer_data_dir', type=str, default='previous_answer_data/')
    parser.add_argument('--error_data_dir', type=str, default='error_data/')
    parser.add_argument('--retrieval_method', type=str, default='e5')
    parser.add_argument('--retrieval_model_path', type=str, default='intfloat/e5-base-v2')
    parser.add_argument('--config_path', type=str, default='myconfig.yaml')
    parser.add_argument('--save_note', type=str)
    parser.add_argument('--dataset_name', type=str, default='nq')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    run(
        dataset_name=args.dataset_name,
        split=args.split,
        model_name=args.model_name,
        plan_model_name=args.plan_model_name,
        execute_model=args.model_name,
        execute_model_path=args.model_path,
        test_sample_num=args.sample_num,
        save_note=args.save_note,
        save_dir=args.save_dir,
        plan_data_dir=args.plan_data_dir,
        retrieval_data_dir=args.retrieval_data_dir,
        previous_answer_data_dir=args.previous_answer_data_dir,
        error_data_dir=args.error_data_dir,
        gpu_id=args.gpu_id,
        retrieval_method=args.retrieval_method,
        retrieval_model_path=args.retrieval_model_path,
        config_path=args.config_path
    )
