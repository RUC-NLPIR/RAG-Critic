<p align="center">
<img src="https://github.com/user-attachments/assets/ec1bad0f-c28a-4238-83b8-cc9b487a6790" alt="rag-critic" style="width: 70%; height: auto;">
</p>


### <div align="center">RAG-Critic: Leveraging Automated Critic-Guided Agentic Workflow for Retrieval Augmented Generation<div>



<div align="center">
<a href= target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv height=25px></a>
<a href=https://huggingface.co/datasets/dongguanting/RAG-Error-Critic-100K target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20Space-276cb4.svg height=25px></a>
<a href=https://colab.research.google.com/drive/1D2myS9UF241gx1otp-fy-LRakMZlROCT?usp=sharing target="_blank"><img src= https://img.shields.io/badge/Google%20Colab-8f2628.svg?logo=googlecolab height=25px></a>
<a href=https://qy-h00.github.io/attention-interpolation-diffusion target="_blank"><img src= https://img.shields.io/badge/GitHub%20Project%20Page-bb8a2e.svg?logo=github height=25px></a>
</div>

<p align="center">
  <br>
  <a href="https://scholar.google.com/citations?user=amozZDkAAAAJ&hl=zh-CN" target="_blank">Guanting Dong</a>,&nbsp;
  <a href="https://scholar.google.com/citations?user=iHCuEo8AAAAJ&hl=zh-CN" target="_blank">Jiajie Jin</a>,&nbsp;
  <a href="https://scholar.google.com/citations?user=XDljV4YAAAAJ&hl=zh-CN" target="_blank">Xiaoxi Li</a>,&nbsp;
  <a href="https://scholar.google.com/citations?user=tBqVOWsAAAAJ&hl=zh-CN" target="_blank">Yutao Zhu</a>,&nbsp;
  <a href="http://playbigdata.ruc.edu.cn/dou/" target="_blank">Zhicheng Dou</a>&nbsp;<sup>&#x2709;</sup>;
  <a href="https://scholar.google.com/citations?user=tbxCHJgAAAAJ&hl=en" target="_blank">Ji-rong Wen</a>
  <br>
</p>

<p align="center">
  Gaoling School of Artificial Intelligence, Renmin University of China.
  <br>
  <sup>&#x2709;</sup> Corresponding Author
</p>

## 📌 Release

[03/2025] We add RAG-Critic demo to further improve clearness, try `demo.ipynb`!

[03/2025] We release our huggingface dataset [🤗RAG-Error-Critic-100K](https://huggingface.co/datasets/dongguanting/RAG-Error-Critic-100K) and Critic model [🤗RAG-Critic-3B](https://huggingface.co/dongguanting/RAG-Critic-3B)

[03/2025] Code and paper are publicly available.


## 🔧 1. General Setup

### Dependencies

- Python 3.10.13
- PyTorch (currently tested on version 2.5.1+cu124)
- Transformers (version 4.47.1, unlikely to work below this version)
- vLLM (version 0.6.6.post1)

```bash
pip install -r requirements.txt
```

### 📝 2. Dataset Preparation
Since the Critic Agent is built on [FlashRAG framework](https://github.com/RUC-NLPIR/FlashRAG), you need to install FlashRAG first.

```bash
# install flashrag
pip install flashrag-dev[full] --pre
# install faiss
conda install -c pytorch faiss-cpu=1.8.0
```

Retrieve the top relevant Wikipedia passages using [E5-base-v2](https://arxiv.org/abs/2212.03533) for 9 RAG-related datasets, stored in the `./dataset_pool_retrieved_top10/${name}` directory. You can find the `train/dev/test` sets of preprocessed datasets with the top 5 retrieved passages [here](https://drive.google.com/drive/folders/1qeLQh8IY173MCXga-oHyuwv8Qw2cb0Jf?usp=sharing). We specify ${dataset} for 9 datasets: ['nq', 'triviaqa', 'hotpotqa', '2wikimultihopqa', 'wikiasp', 'eli5', 'asqa', 'fever', 'wow'] in the following example commands.





## Hierarchical Error System Construction

We design a three-step pipeline for error response mining and annotation, establishing a hierarchical RAG error categorization system.

### Overview  

<img width="868" alt="image" src="https://github.com/user-attachments/assets/596eb45d-9193-4f8e-9f4e-e7911d2c2acd" />

As shown in the image above, we have a total of 7 first-tier labels, 19 second-tier labels, and over 4000 third-tier labels. Here are the details:

1. **Hierarchical Error System** ([here](https://github.com/dongguanting/RAG-Critic/blob/main/all_tags_structure_final.json)).
2. **Frequency error statistics of labels** ([here](https://github.com/dongguanting/RAG-Critic/blob/main/error_tag_frequent.txt)).

<details>
<summary>🔍 Click here! If you want to reproduce our RAG error response mining and annotation.</summary>

### Step 1: Error Response Sampling
First, please download the sampling models from Hugging Face (refer to Appendix Table 9 -- 15 models), and place these model names in the models parameter. Then, perform comprehensive response sampling on the 9 RAG-related datasets:
```bash
cd ./error_system_construction/
bash sample.sh
```
The output data will be saved at `error_sampling_results/responses_${model}_${dataset}_train_1w.json`.

### Step 2: Critical Annotation & Tagging

1. **Critical Annotation**  
   Analyze the reasons for errors using the strong supervision model (Qwen2.5-72B) on the sampled data containing Chain of Thought responses. First, download the sampling models from Hugging Face (refer to Appendix Table 9 -- 15 models), then perform comprehensive response sampling on the 9 RAG-related datasets:
   ```bash
   cd ./error_system_construction/
   bash critic.sh
   ```
   The source data and error analysis will be saved at `error_critic_results/critic_${model}_${dataset}_train_1w.json`.

2. **Tagging**  
   Inspired by the Instag prompt template, we further annotate the RAG error analysis results with fine-grained, open-set labels:
   ```bash
   cd ./error_system_construction/
   bash error_tag.sh
   ```
   The sampled open-set tags will be saved at `error_critic_results/critic_${model}_${dataset}_train_1w.json`.

### Step 3: Error Label Summarization

First, please follow the methods in the document to deduplicate and normalize the tag set. Then, refer to the hierarchical clustering method for aggregating RAG error clusters, as detailed in [cluster.ipynb](https://github.com/dongguanting/RAG-Critic/blob/main/error_system_construction/cluster.ipynb). 

Furthermore, use GPT-4-o and human input for higher-level label summarization of the error clusters.

</details>

## 🚗 RAG Error-Critic Alignment

We release our RAG Error-Critic SFT dataset and model weights:

- **SFT Dataset:** We synthesize the first fine-grained error identification dataset, [🤗RAG-Error-Critic-100K](https://huggingface.co/datasets/dongguanting/RAG-Error-Critic-100K), by combining responses from 15 models across 9 RAG-related datasets with fine-grained error labels.

- **Model Weights:** We released our RAG error identification model [🤗RAG-Critic-3B](https://huggingface.co/dongguanting/RAG-Critic-3B) for fine-grained error recognition.

The following shows our detailed training procedure:

- **SFT bash:**
  
```bash
### model
model_name_or_path: /path/to/model_zoo/model_name

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: /path/to/deepspeed/config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

### dataset
dataset: dataset_name
template: template_name
cutoff_len: 4096
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: /path/to/output/directory
logging_steps: 10
save_steps: 2000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

---

For DPO data, please construct it based on our SFT dataset and error system settings (Section 3.2), using the previous version of [LlaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/releases/tag/v0.6.3).

- **Coarse-to-Fine DPO bash:**

```bash
deepspeed --num_gpus 8 train_bash.py \
        --deepspeed $deepspeed_zero3_config_path \
        --stage dpo \
        --do_train \
        --model_name_or_path $MODEL_PATH \
        --dataset $dataset \
        --dataset_dir $DATA_PATH \
        --template $Template \
        --finetuning_type full \
        --output_dir $OUTPUT_PATH \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len 4096 \
        --preprocessing_num_workers 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --warmup_ratio 0.1 \
        --save_steps 1000 \
        --learning_rate 5e-6 \
        --num_train_epochs 2.0 \
        --max_samples 200000 \
        --ddp_timeout 180000000 \
        --plot_loss \
        --fp16
```


## Critic-Guiided Agentic Workflow

### Data Preparation

<details>
<summary>🔍 Click here to generate the test set.</summary>

Firstly, using critic agent to obtain required correction path

**Step 1: Install Required Frameworks**

Since the Critic Agent is built on [FlashRAG framework](https://github.com/RUC-NLPIR/FlashRAG), you need to install FlashRAG first.

```bash
# install flashrag
pip install flashrag-dev[full] --pre
# install faiss
conda install -c pytorch faiss-cpu=1.8.0
```

**Step 2: Prepare the Data**

The operation of Critic requires the following data:

1. Retrieved Documents: These contain the retrieval results for each query in the test set (used to generate the original answer retrieval results). The storage path is: `{retrieval_data_dir}/{dataset_name}/{split}.json`. The format is as follows:

```json
[
    {
        "question": "who sings does he love me with reba",
        "golden_answers": [
            "Linda Davis"
        ],
        "retrieval_docs": [
            {
                "id": "17237290",
                "contents": "\"Does He Love You\"\nDoes He Love You \"\"Does He Love You\"\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album \"\"Greatest Hits Volume Two\"\". It is one of country music's several songs about a love triangle. \"\"Does He Love You\"\" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members"
            },
            {
                "id": "5026369",
                "contents": "\"Linda Davis\"\nLinda Davis Linda Kaye Davis (born November 26, 1962) is an American country music singer. Before beginning a career as a solo artist, she had three minor country singles in the charts as one half of the duo Skip & Linda. In her solo career, Davis has recorded five studio albums for major record labels and more than 15 singles. Her highest chart entry is \"\"Does He Love You\"\", her 1993 duet with Reba McEntire, which reached number one on the \"\"Billboard\"\" country charts and won both singers the Grammy for Best Country Vocal Collaboration. Her highest solo chart position"
            }
        ]
    }
]
```

2. The raw answers file generated by the model, stored at: `{previous_answer_data_dir}/responses_{model_name}_{dataset_name}_{split}_100.json`
3. The error analysis file annotated by the Critic model, stored at: `error_data_dir/errordata_{dataset_name}_{model_name}_{split}.json`

The generation of the last two files can refer to the steps for generating model raw answers and running Critic.

**Step 3: Prepare the Retriever**

Since the Agent needs to perform retrieval during its execution, you need to download the retrieval Corpus and its corresponding Index. The experiment uses the Wiki-dpr-100w file and the corresponding E5 index provided by FlashRAG. The download links are as follows:
- [https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/resolve/master/retrieval_corpus/wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/resolve/master/retrieval_corpus/wiki18_100w_e5_index.zip)
- [https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/resolve/master/retrieval_corpus/wiki18_100w.jsonl](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/resolve/master/retrieval_corpus/wiki18_100w.jsonl)

**Step 4: Fill in the Configuration File**

After downloading the necessary files, you need to fill in the file paths into the configuration file required by FlashRAG (`myconfig.yaml`). The fields that need to be filled are as follows, with other fields filled during the program execution:

- method2index
- corpus_path
</details>




### Usage

`plan_agent.py` and `execute_agent.py` respectively provide the Critic's planning and execution. The running scripts are in `run_exp.sh`. After running, the evaluation results and intermediate variables will be stored in the corresponding folder under `save_dir`.

You can directly run the following command to execute the Critic Agent:

```bash
cd ./critic_agent/
python run_exp.sh
```


## RAG-Error Bench

We introduce the RAG-Error benchmark, aiming to make prediction judgment and fine-grained error recognition in RAG.


![image](https://github.com/user-attachments/assets/97c9a569-5712-499e-bcc3-d04df06ea307)



### 📊 Test Cases

**Key-Value Introduction:**

- **Input:** including three parts: User query + Top-K Document + LLM's prediction + 1st-tier error tag sets (all) + 2nd-tier erro tag sets (all)
- **Output:** Judgement, 1st-tier error tag sets (selected), 2nd-tier erro tag sets (selected)


```json
{
    "instruction": 
"You are a critical system designed to provide useful error type tags for retrieval-augmented generation (RAG) tasks. Your goal is to assist in detailed error analysis to improve the performance of AI assistants. Below are the [Question], the top-5 retrieved relevant [Passages], and the [Model's Prediction] for the RAG tasks.\n\n                Question: who wrote the song going to kansas city\n                Passage 1: \"Kansas City (Leiber and Stoller song)\"\nKansas City (Leiber and Stoller song) \"\"Kansas City\"\" is a rhythm and blues song written by Jerry Leiber and Mike Stoller in 1952. First recorded by Little Willie Littlefield the same year, the song later became a #1 hit when it was recorded by Wilbert Harrison in 1959. \"\"Kansas City\"\" became one of Leiber and Stoller's \"\"most recorded tunes, with more than three hundred versions,\"\" with several appearing in the R&B and pop record charts. \"\"Kansas City\"\" was written by Jerry Leiber and Mike Stoller, two nineteen-year-old rhythm and blues fans from Los Angeles, who had their first success writing\n                Passage 2: \"Kansas City (Leiber and Stoller song)\"\n\"\"Eighteenth and Vine\"\" for \"\"12th Street and Vine,\"\" which sings just as well, and recognizes Kansas City's jazz history. Kansas City (Leiber and Stoller song) \"\"Kansas City\"\" is a rhythm and blues song written by Jerry Leiber and Mike Stoller in 1952. First recorded by Little Willie Littlefield the same year, the song later became a #1 hit when it was recorded by Wilbert Harrison in 1959. \"\"Kansas City\"\" became one of Leiber and Stoller's \"\"most recorded tunes, with more than three hundred versions,\"\" with several appearing in the R&B and pop record charts. \"\"Kansas City\"\" was written by Jerry\n                Passage 3: \"Going to Kansas City\"\nGoing to Kansas City Going to Kansas City is a 1998 Canadian-Finnish drama film directed by Pekka Mandart. The film is about a male exchange student from Finland, who falls in love with an American girl, whose father does not accept the relationship. Shot in Canada, the film is set in the rural town of Canaan that is located 120 miles west of Kansas City. The director Pekka Mandart adapted the idea of the story from a magazine article about a love story between a Finnish exchange student and an American girl. Out of a total budget of 17 million\n                Passage 4: \"Wilbert Harrison\"\nWilbert Harrison Wilbert Huntington Harrison (January 5, 1929 – October 26, 1994) was an American rhythm and blues singer, pianist, guitarist and harmonica player. Born in Charlotte, North Carolina, Harrison had a Billboard #1 record in 1959 with the song \"\"Kansas City\"\". The song was written in 1952 and was one of the first credited collaborations by the team of Jerry Leiber and Mike Stoller. It sold over one million copies, and was awarded a gold disc. Harrison recorded \"\"Kansas City\"\" for the Harlem-based entrepreneur Bobby Robinson, who released it on his Fury record label. At the height of the\n                Passage 5: \"Wilbert Harrison\"\nthe North Carolina Music Hall of Fame in 2009. Wilbert Harrison Wilbert Huntington Harrison (January 5, 1929 – October 26, 1994) was an American rhythm and blues singer, pianist, guitarist and harmonica player. Born in Charlotte, North Carolina, Harrison had a Billboard #1 record in 1959 with the song \"\"Kansas City\"\". The song was written in 1952 and was one of the first credited collaborations by the team of Jerry Leiber and Mike Stoller. It sold over one million copies, and was awarded a gold disc. Harrison recorded \"\"Kansas City\"\" for the Harlem-based entrepreneur Bobby Robinson, who released it on\n                Model's Prediction: The song \"Kansas City\" was not written by the individuals or entities mentioned in the passages about \"Going to Kansas City,\" \"Wilbert Harrison,\" or \"Kansas City (Leiber and Stoller song).\" The song \"Kansas City\" was written by Jerry Leiber and Mike Stoller in 1952.\n\n                Please first determine whether the model's prediction is correct. If it is correct, output it as follows:\n                {\"Judgement\": \"Correct\"}\n\n                If it is incorrect, give these error types, tag1 corresponds to tag2 one-to-one：\n                        tag1= [\n                                \"Incomplete Information\",\n                                \"Incomplete Information\",\n                                \"Irrelevant Information\",\n                                \"Irrelevant Information\",\n                                \"Erroneous Information\",\n                                \"Incomplete or Missing Response\",\n                                \"Incomplete or Missing Response\",\n                                \"Incomplete or Missing Response\",\n                                \"Incomplete or Missing Response\",\n                                \"Incomplete or Missing Response\",\n                                \"Inaccurate or Misunderstood Response\",\n                                \"Inaccurate or Misunderstood Response\",\n                                \"Inaccurate or Misunderstood Response\",\n                                \"Inaccurate or Misunderstood Response\",\n                                \"Inaccurate or Misunderstood Response\",\n                                \"Inaccurate or Misunderstood Response\",\n                                \"Irrelevant or Off-Topic Response\",\n                                \"Irrelevant or Off-Topic Response\",\n                                \"Irrelevant or Off-Topic Response\",\n                                \"Overly Verbose Response\"\n                            ]\n\n                            tag2 = [\n                                \"Insufficient or Incomplete Information Retrieval\",\n                                \"Data Insufficiency in Retrieval\",\n                                \"Relevance Gaps in Retrieval\",\n                                \"Irrelevant Information Retrieval\",\n                                \"Erroneous Information Retrieval\",\n                                \"Omission of Key Information\",\n                                \"Lack of Specificity\",\n                                \"Specificity and Precision Errors\",\n                                \"Partial Coverage and Temporal Issues\",\n                                \"Lack of Practicality\",\n                                \"Contextual Understanding Errors\",\n                                \"Factual Inaccuracies\",\n                                \"Incorrect and Incomplete Answers\",\n                                \"Golden Answer Misalignment\",\n                                \"Misinterpretation of Queries and Information\",\n                                \"Entity and Concept Confusion\",\n                                \"Irrelevant Content and Topic Drift\",\n                                \"Off-Topic and Redundant Responses\",\n                                \"Content and Context Misalignment\",\n                                \"Overly Complex and Redundant Response\"\n                            ]\n                 Please identify the error tags at three levels, from coarse to fine, and provide a detailed error analysis. Adhere strictly to the following JSON format:\n                {\n                    \"Judgement\": \"Error\",\n                    \"Error_analysis\": \"\",\n                    \"tag1\": [],\n                    \"tag2\": [],\n                    \"tag3\": []\n                }\n                ",
    "input": "",
    "output": "{'Judgement': 'Error', 'Error_analysis': \"The model's prediction is incorrect because it does not directly answer the question about who wrote the song 'Going to Kansas City.' Instead, it provides information about the song 'Kansas City,' which is a different song. The passages provided do not contain information about the song 'Going to Kansas City,' and thus the model should have indicated that the information to answer the question is not available in the provided passages.\", 'tag1': ['Incomplete or Missing Response', 'Inaccurate or Misunderstood Response', 'Incomplete Information'], 'tag2': ['Entity and Concept Confusion', 'Lack of Specificity', 'Insufficient or Incomplete Information Retrieval', 'Contextual Understanding Errors'], 'tag3': ['Relevance Error', 'Contextual Understanding Error', 'Information Retrieval Failure', 'Specificity Error']}",
    "history": [

    ]
}
```




### 🔑 Inference
You first need to perform inference on RAG-Error bench, and the command is as follows:
```bash
cd ./rag_error_bench/

# Evaluate open-sourced LLM
bash test_open_llm.sh

# Evaluate close-source LLM like GPT-4o, Deepseek R1 and Claude 3.5
bash test_close_llm.sh 
```

The format of each sample in your ‘RAG-Critic/rag_error_bench/test_data/baseline_test.json’ are in the following form:
<img width="614" alt="image" src="https://github.com/user-attachments/assets/4c19a234-c6d1-415c-9a4a-2e34afefafa5" />



### 📝 Evaluation
After completing the inference, run the evaluation script:

```bash
python ./rag_error_bench/caculate_acc.py
```

Note that you need to replace the input and output sections of 'caculate_acc.py'.

<details>
<summary>🔍 Here, we provide detailed evaluation metric results of the RAG-Error bench in the following format.</summary>

```json
{
  "overall": {
    "accuracy": 0.1194,  #Overall Acc.
    "f1": 0.1781,    #Overall F1
    "rouge": {
      "rouge-1": 0.4707,
      "rouge-2": 0.2361,
      "rouge-l": 0.4359
    },
    "judgement_accuracy": 0.6895,  #Overall Judgment
    "correct_judgement_accuracy": 0.9526,   #Overall judgement of correct prediction
    "tag1": {
      "accuracy": 0.1741,  #Overall Tag1 Acc.
      "f1": 0.2567,    #Overall Tag2 F1 Acc.
      "rouge": {
        "rouge-1": 0.4707,
        "rouge-2": 0.2361,
        "rouge-l": 0.4359
      },
      "judgement_accuracy": 0.4221   #Overall judgement of error prediction
    },
    "tag2": {
      "accuracy": 0.0647,   #Overall Tag2 Acc.
      "f1": 0.0995,    #Overall Tag2 F1 Acc.
      "rouge": {
        "rouge-1": 0.4707,
        "rouge-2": 0.2361,
        "rouge-l": 0.4359
      },
      "judgement_accuracy": 0.4221
    }
  },
  "category_metrics": {    #Coarse-grained Error Tags Acc.
    "tag1": {
      "Incomplete Information": {
        "accuracy": 0.2077,
        "f1": 0.2961,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Irrelevant Information": {
        "accuracy": 0.1289,
        "f1": 0.1968,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Erroneous Information": {
        "accuracy": 0.036,
        "f1": 0.0541,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Incomplete or Missing Response": {
        "accuracy": 0.1618,
        "f1": 0.2585,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Inaccurate or Misunderstood Response": {
        "accuracy": 0.273,
        "f1": 0.3803,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Irrelevant or Off-Topic Response": {
        "accuracy": 0.0103,
        "f1": 0.0188,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Overly Verbose Response": {
        "accuracy": 0.4259,
        "f1": 0.2771,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      }
    },
    "tag2": {
      "Insufficient or Incomplete Information Retrieval": {
        "accuracy": 0.2028,
        "f1": 0.2677,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Data Insufficiency in Retrieval": {
        "accuracy": 0.0053,
        "f1": 0.0106,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Relevance Gaps in Retrieval": {
        "accuracy": 0.2483,
        "f1": 0.2483,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Irrelevant Information Retrieval": {
        "accuracy": 0.0,
        "f1": 0,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Erroneous Information Retrieval": {
        "accuracy": 0.036,
        "f1": 0.0543,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Omission of Key Information": {
        "accuracy": 0.1565,
        "f1": 0.1513,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Lack of Specificity": {
        "accuracy": 0.0,
        "f1": 0,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Specificity and Precision Errors": {
        "accuracy": 0.0,
        "f1": 0,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Partial Coverage and Temporal Issues": {
        "accuracy": 0.0,
        "f1": 0,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Lack of Practicality": {
        "accuracy": 0.0,
        "f1": 0,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Contextual Understanding Errors": {
        "accuracy": 0.1971,
        "f1": 0.1843,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Factual Inaccuracies": {
        "accuracy": 0.0186,
        "f1": 0.0348,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Incorrect and Incomplete Answers": {
        "accuracy": 0.0073,
        "f1": 0.0143,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Misinterpretation of Queries and Information": {
        "accuracy": 0.0693,
        "f1": 0.0676,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Entity and Concept Confusion": {
        "accuracy": 0.0089,
        "f1": 0.0171,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Irrelevant Content and Topic Drift": {
        "accuracy": 0.0125,
        "f1": 0.0185,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Off-Topic and Redundant Responses": {
        "accuracy": 0.0,
        "f1": 0,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Content and Context Misalignment": {
        "accuracy": 0.0,
        "f1": 0,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      },
      "Overly Complex and Redundant Response": {
        "accuracy": 0.4259,
        "f1": 0.2788,
        "rouge": {
          "rouge-1": 0.4707,
          "rouge-2": 0.2361,
          "rouge-l": 0.4359
        }
      }
    }
  }
}
```

</details>


## 📜 License

Our dataset are distributed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.



## 🎖 Citation 

Please cite our work if you find the repository helpful.

```

```










## ❤️ Acknowledgement

We thank the following repositories for their great work: [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG), [transformers](https://github.com/huggingface/transformers), [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) and [FollowRAG](https://github.com/dongguanting/FollowRAG).
