

<p align="center">
  <img src="sdxl_results/aesthetic/1.jpg">
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
[10/2024] We are now supporting various application including compositional generation, image morphing, image editing and image-control generation (based on IP-Adapter), try `play_sdxl_application.ipynb` and `play_sd.ipynb`. It's also available in our Hugging Face Space [AID-v2](https://huggingface.co/spaces/qyoo/AID-v2). Have fun!

[10/2024] We are now supporting interpolating between images via [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)!

[10/2024] We add dynamic selection pipeline to further improve smoothness, try `play_sdxl_trial.ipynb`!

[10/2024] PAID is accepted as a conference paper by NeurIPS 2024!

[03/2025] Code and paper are publicly available.





---

## 🌠 VIF-RAG


We broke down the VIF-RAG data synthesis process into steps and provided 10-20 samples for each step to assist with your reproduction. Be sure to replace these with your own input.

<img width="1243" alt="image" src="https://github.com/user-attachments/assets/d38871d3-d29d-425b-a7d5-d8a7081a110d">



### :wrench: Dependencies
General Setup Environment:
- Python 3.9
- [PyTorch](http://pytorch.org/) (currently tested on version 2.1.2+cu121)
- [Transformers](http://huggingface.co/transformers/) (version 4.41.2, unlikely to work lower than this version)

```bash
cd ./VIF-RAG/
pip install -r requirements.txt
cd ./FollowRAG/
pip install -r requirements.txt
```

### :rocket: How to Perform *VIF-RAG* Data Synthesis?


Follow the interactive Jupyter notebook VIF-RAG on ``vifrag.ipynb`` to reproduce our synthesize dataset.


### 🎯 Training

We use the version of [LlaMA-Factory v0.6.3](https://github.com/hiyouga/LLaMA-Factory/releases/tag/v0.6.3). Thanks for their excellent work.

we also release our SFT version dataset as strong baseline in Table1:
- **SFT Version:** To make a fair comparison with VIF-RAG, we use the same amount of [🤗ShareGPT](https://huggingface.co/datasets/dongguanting/ShareGPT-12K) and [🤗RAG-QA-40K](https://huggingface.co/datasets/dongguanting/RAG-QA-40K) as in VIF-RAG’s data synthesis process, mixing them together to fine-tune (SFT) different baseline models.

- **VIF-RAG-QA:** We release our SFT datasets, including [🤗VIF-RAG-QA-110K](https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-110K) and [🤗VIF-RAG-QA-20K](https://huggingface.co/datasets/dongguanting/VIF-RAG-QA-20K).


- **SFT bash:**
  
```bash
deepspeed --num_gpus=8 train_bash.py \
        --deepspeed $deepspeed_zero3_config_path \
        --stage sft \
        --do_train \
        --use_fast_tokenizer \
        --flash_attn \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --model_name_or_path $MODEL_PATH \
        --dataset $dataset \
        --template $Template \
        --finetuning_type full \
        --output_dir $OUTPUT_PATH \
        --overwrite_cache \
        --overwrite_output_dir \
        --warmup_steps 20 \
        --weight_decay 0.1 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --ddp_timeout 9000 \
        --learning_rate 7e-6 \
        --lr_scheduler_type "linear" \
        --logging_steps 1 \
        --cutoff_len 8192 \
        --save_steps 200 \
        --num_train_epochs 3.0 \
        --plot_loss \
        --bf16 
```

---

## 🐋 FollowRAG 

FollowRAG is the first benchmark designed to comprehensively evaluate LLM’s complex instruction-following abilities in RAG. 

<img width="1070" alt="image" src="https://github.com/user-attachments/assets/91a5e7ac-d828-46f2-bcae-96886f7ef295">


### 📊 Test Cases

<details>
<summary>🔍 Click here! if you are curious about FollowRAG‘s test cases.</summary>

**Key-Value Introduction:**

- **prompt:** The complete question for FollowRAG, including three parts: TopK Document + user query + instruction
- **question:** QA question (sourced from NQ)
- **answer_gold:** Reference answer (note that this is not the golden answer, as the answer needs to follow instruction constraints after adding instructions)
- **question_with_instrs:** QA question + a series of instruction constraints
- **instruction_id_list & kwargs:** Instruction types and parameters needed for evaluation calculation
- **passages:** TopK documents retrieved from Wiki using DPR



```json

    {
        "key": 0,
        "type": "ifnq",
        "prompt": "Given the following information: \nPassage-0 Title: Gravity Content: and prevents further acceleration. The force of gravity on Earth is the resultant (vector sum) of two forces: (a) The gravitational attraction in accordance with Newton's universal law of gravitation, and (b) the centrifugal force, which results from the choice of an earthbound, rotating frame of reference. The force of gravity is the weakest at the equator because of the centrifugal force caused by the Earth's rotation and because points on the equator are furthest from the center of the Earth. The force of gravity varies with latitude and increases from about 9.780 m/s at the Equator to about 9.832\nPassage-1 Title: Gravitational acceleration Content: Gravitational acceleration In physics, gravitational acceleration is the acceleration on an object caused by the force of gravitation. Neglecting friction such as air resistance, all small bodies accelerate in a gravitational field at the same rate relative to the center of mass. This equality is true regardless of the masses or compositions of the bodies. At different points on Earth, objects fall with an acceleration between and depending on altitude and latitude, with a conventional standard value of exactly 9.80665 m/s (approximately 32.174 ft/s). This does not take into account other effects, such as buoyancy or drag. Newton's law of\nPassage-2 Title: Gravity Content: Gravity Gravity (), or gravitation, is a natural phenomenon by which all things with mass or energy—including planets, stars, galaxies, and even light—are brought toward (or \"gravitate\" toward) one another. On Earth, gravity gives weight to physical objects, and the Moon's gravity causes the ocean tides. The gravitational attraction of the original gaseous matter present in the Universe caused it to begin coalescing, forming starsand for the stars to group together into galaxiesso gravity is responsible for many of the large-scale structures in the Universe. Gravity has an infinite range, although its effects become increasingly weaker on farther objects. Gravity\n\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source.\nQuestion: What is the common name for gravitational force? In this task, repeat the exact request first, then give your response. Do not say any word before repeating the exact request. Moreover, your answer must contain a title, wrapped in double angular brackets, i.e. <<title>>. Ensure the word disappointed appears at least twice. Finally, provide your answer with less than 200 words.",
        "question": "what is the common name for gravitational force",
        "answer_gold": "Gravity/Gravity, or gravitation",
        "question_with_instrs": "What is the common name for gravitational force? In this task, repeat the exact request first, then give your response. Do not say any word before repeating the exact request. Moreover, your answer must contain a title, wrapped in double angular brackets, i.e. <<title>>. Ensure the word disappointed appears at least twice. Finally, provide your answer with less than 200 words.",
        "instruction_id_list": [
            "combination:repeat_prompt",
            "detectable_format:title",
            "keywords:frequency",
            "length_constraints:number_words"
        ],
        "kwargs": [
            {
                "prompt_to_repeat": "What is the common name for gravitational force?"
            },
            {},
            {
                "relation": "at least",
                "keyword": "disappointed",
                "frequency": 2
            },
            {
                "relation": "less than",
                "num_words": 200
            }
        ],
        "passages": [
            {
                "title": "Gravity",
                "content": "and prevents further acceleration. The force of gravity on Earth is the resultant (vector sum) of two forces: (a) The gravitational attraction in accordance with Newton's universal law of gravitation, and (b) the centrifugal force, which results from the choice of an earthbound, rotating frame of reference. The force of gravity is the weakest at the equator because of the centrifugal force caused by the Earth's rotation and because points on the equator are furthest from the center of the Earth. The force of gravity varies with latitude and increases from about 9.780 m/s at the Equator to about 9.832"
            },
            {
                "title": "Gravitational acceleration",
                "content": "Gravitational acceleration In physics, gravitational acceleration is the acceleration on an object caused by the force of gravitation. Neglecting friction such as air resistance, all small bodies accelerate in a gravitational field at the same rate relative to the center of mass. This equality is true regardless of the masses or compositions of the bodies. At different points on Earth, objects fall with an acceleration between and depending on altitude and latitude, with a conventional standard value of exactly 9.80665 m/s (approximately 32.174 ft/s). This does not take into account other effects, such as buoyancy or drag. Newton's law of"
            },
            {
                "title": "Gravity",
                "content": "Gravity Gravity (), or gravitation, is a natural phenomenon by which all things with mass or energy—including planets, stars, galaxies, and even light—are brought toward (or \"gravitate\" toward) one another. On Earth, gravity gives weight to physical objects, and the Moon's gravity causes the ocean tides. The gravitational attraction of the original gaseous matter present in the Universe caused it to begin coalescing, forming starsand for the stars to group together into galaxiesso gravity is responsible for many of the large-scale structures in the Universe. Gravity has an infinite range, although its effects become increasingly weaker on farther objects. Gravity"
            }
        ]
    }
```
</details>




### 🔑 Inference
You first need to perform inference on followRAG, and the pseudocode is as follows:
```python
followRAG_full=load_json('followRAG/followRAG_full.json')
data_inferenced=[]
for dp in followRAG_full:
    response=llm.inference(dp['prompt'])
    dp['response']=response
    data_inferenced.append(dp)
save_jsonl(data_inferenced,'results/finish_inference/data_inferenced.jsonl')
```
Please refer to the following template to prepare your result JSON files for subsequent evaluation. 
The format of each sample in your data_inferenced.jsonl should be consistent with the following form:
```json

{
    "key": 0,
    "type": "ifnq",
    "prompt": "Given the following information: \nPassage-0 Title: Gravity Content: and prevents further acceleration. The force of gravity on Earth is the resultant (vector sum) of two forces: (a) The gravitational attraction in accordance with Newton's universal law of gravitation, and (b) the centrifugal force, which results from the choice of an earthbound, rotating frame of reference. The force of gravity is the weakest at the equator because of the centrifugal force caused by the Earth's rotation and because points on the equator are furthest from the center of the Earth. The force of gravity varies with latitude and increases from about 9.780 m/s at the Equator to about 9.832\nPassage-1 Title: Gravitational acceleration Content: Gravitational acceleration In physics, gravitational acceleration is the acceleration on an object caused by the force of gravitation. Neglecting friction such as air resistance, all small bodies accelerate in a gravitational field at the same rate relative to the center of mass. This equality is true regardless of the masses or compositions of the bodies. At different points on Earth, objects fall with an acceleration between and depending on altitude and latitude, with a conventional standard value of exactly 9.80665 m/s (approximately 32.174 ft/s). This does not take into account other effects, such as buoyancy or drag. Newton's law of\nPassage-2 Title: Gravity Content: Gravity Gravity (), or gravitation, is a natural phenomenon by which all things with mass or energy—including planets, stars, galaxies, and even light—are brought toward (or \"gravitate\" toward) one another. On Earth, gravity gives weight to physical objects, and the Moon's gravity causes the ocean tides. The gravitational attraction of the original gaseous matter present in the Universe caused it to begin coalescing, forming starsand for the stars to group together into galaxiesso gravity is responsible for many of the large-scale structures in the Universe. Gravity has an infinite range, although its effects become increasingly weaker on farther objects. Gravity\n\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source.\nQuestion: What is the common name for gravitational force? In this task, repeat the exact request first, then give your response. Do not say any word before repeating the exact request. Moreover, your answer must contain a title, wrapped in double angular brackets, i.e. <<title>>. Ensure the word disappointed appears at least twice. Finally, provide your answer with less than 200 words.",
    "question": "what is the common name for gravitational force",
    "answer_gold": "Gravity/Gravity, or gravitation",
    "question_with_instrs": "What is the common name for gravitational force? In this task, repeat the exact request first, then give your response. Do not say any word before repeating the exact request. Moreover, your answer must contain a title, wrapped in double angular brackets, i.e. <<title>>. Ensure the word disappointed appears at least twice. Finally, provide your answer with less than 200 words.",
    "instruction_id_list": [
        "combination:repeat_prompt",
        "detectable_format:title",
        "keywords:frequency",
        "length_constraints:number_words"
    ],
    "kwargs": [
        {
            "prompt_to_repeat": "What is the common name for gravitational force?"
        },
        {

        },
        {
            "relation": "at least",
            "keyword": "disappointed",
            "frequency": 2
        },
        {
            "relation": "less than",
            "num_words": 200
        }
    ],
    "passages": [
        {
            "title": "Gravity",
            "content": "and prevents further acceleration. The force of gravity on Earth is the resultant (vector sum) of two forces: (a) The gravitational attraction in accordance with Newton's universal law of gravitation, and (b) the centrifugal force, which results from the choice of an earthbound, rotating frame of reference. The force of gravity is the weakest at the equator because of the centrifugal force caused by the Earth's rotation and because points on the equator are furthest from the center of the Earth. The force of gravity varies with latitude and increases from about 9.780 m/s at the Equator to about 9.832"
        },
        {
            "title": "Gravitational acceleration",
            "content": "Gravitational acceleration In physics, gravitational acceleration is the acceleration on an object caused by the force of gravitation. Neglecting friction such as air resistance, all small bodies accelerate in a gravitational field at the same rate relative to the center of mass. This equality is true regardless of the masses or compositions of the bodies. At different points on Earth, objects fall with an acceleration between and depending on altitude and latitude, with a conventional standard value of exactly 9.80665 m/s (approximately 32.174 ft/s). This does not take into account other effects, such as buoyancy or drag. Newton's law of"
        },
        {
            "title": "Gravity",
            "content": "Gravity Gravity (), or gravitation, is a natural phenomenon by which all things with mass or energy—including planets, stars, galaxies, and even light—are brought toward (or \"gravitate\" toward) one another. On Earth, gravity gives weight to physical objects, and the Moon's gravity causes the ocean tides. The gravitational attraction of the original gaseous matter present in the Universe caused it to begin coalescing, forming starsand for the stars to group together into galaxiesso gravity is responsible for many of the large-scale structures in the Universe. Gravity has an infinite range, although its effects become increasingly weaker on farther objects. Gravity"
        }
    ],
    "response": "What is the common name for gravitational force? \n\n<<Gravity>>\n\nThe common name for gravitational force is \"gravity.\" It is often disappointing to realize how misunderstood this fundamental force can be, as people frequently overlook its complexities. Many assume that gravity acts uniformly across the globe, but the actual gravitational acceleration varies depending on location—grounded by factors such as latitude and altitude. \n\nFor instance, a commonly cited value is 9.80665 m/s², yet we can be disappointed to discover that at the equator, gravity is slightly weaker due to centrifugal forces caused by the Earth's rotation. This leads to significant implications in both physics and everyday life, yet many remain uninformed about these nuances. \n\nUnderstanding gravity is essential not only for grasping the movement of objects on Earth but also for comprehending cosmic phenomena. It is disappointing when people take gravity for granted, missing the fascinating science behind how it governs numerous aspects of our universe."
}

```


### 📝 Evaluation
After completing the inference, run the evaluation script:
```bash
export OPENAI_API_KEY=your_openai_api_key
export OPENAI_API_BASE=https://api.openai.com/v1
python eval/main_eval.py \
    --input_file_path results/finish_inference/data_inferenced.jsonl \
    --output_file_path results/finish_eval/data_evaled.jsonl \
    --rag_eval_type mini \
    --result_log_file_path results/logs/results_log.jsonl
```
Our evaluation of instruction-following part largely draws on the [IFEval code repository](https://github.com/google-research/google-research/tree/master/instruction_following_eval). We appreciate their excellent work!


## 📜 License

Our dataset are distributed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.



## 🎖 Citation 

Please cite our work if you find the repository helpful.

```
@article{dong2024general,
  author       = {Guanting Dong and
                  Xiaoshuai Song and
                  Yutao Zhu and
                  Runqi Qiao and
                  Zhicheng Dou and
                  Ji{-}Rong Wen},
  title        = {Toward General Instruction-Following Alignment for Retrieval-Augmented
                  Generation},
  journal      = {CoRR},
  volume       = {abs/2410.09584},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2410.09584},
  doi          = {10.48550/ARXIV.2410.09584},
  eprinttype    = {arXiv},
  eprint       = {2410.09584},
  timestamp    = {Fri, 22 Nov 2024 21:38:25 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2410-09584.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```














## 📑 Abstract

<b>TL;DR: <font color="red">AID</font> (Attention Interpolation via Diffusion)</b> is a training-free method that enables the text-to-image diffusion model to generate interpolation between different conditions with high consistency, smoothness and fidelity. Its variant, <font color="blue">PAID</font>, provides further control of the interpolation via prompt guidance.

<details><summary>CLICK for the full abstract</summary>
Conditional diffusion models can create unseen images in various settings, aiding image interpolation. Interpolation in latent spaces is well-studied, but interpolation with specific conditions like text or poses is less understood. Simple approaches, such as linear interpolation in the space of conditions, often result in images that lack consistency, smoothness, and fidelity. To that end, we introduce a novel training-free technique named Attention Interpolation via Diffusion (AID). Our key contributions include 1) proposing an inner/outer interpolated attention layer; 2) fusing the interpolated attention with self-attention to boost fidelity; and 3) applying beta distribution to selection to increase smoothness. We also present a variant, Prompt-guided Attention Interpolation via Diffusion (PAID), that considers interpolation as a condition-dependent generative process. This method enables the creation of new images with greater consistency, smoothness, and efficiency, and offers control over the exact path of interpolation. Our approach demonstrates effectiveness for conceptual and spatial interpolation.
</details>

## ▶️ PAID Results

<p align="center">
<img src="sdxl_results/aesthetic/3.jpg">
</p>

<p align="center">
<img src="sdxl_results/anime/3.jpg">
</p>

<p align="center">
<img src="sdxl_results/photorealistic/1.jpg">
</p>

<details><summary>CLICK for more results </summary>

#### Aesthetic

<p align="center">
<img src="sdxl_results/aesthetic/2.jpg">
</p>

<p align="center">
<img src="sdxl_results/aesthetic/4.jpg">
</p>

#### Anime

<p align="center">
<img src="sdxl_results/anime/1.jpg">
</p>

<p align="center">
<img src="sdxl_results/anime/2.jpg">
</p>

#### Photorealistic

<p align="center">
<img src="sdxl_results/photorealistic/2.jpg">
</p>

<p align="center">
<img src="sdxl_results/photorealistic/3.jpg">
</p>

</details>

## 📷 Application

<p align="center">
<img src="asset/applications.png">
</p>

### Compositional Generation

Given a prompt that involves multiple components (e.g., "A dog driving a car"), we use the compositional description as a guidance prompt, with each related component (e.g., "A dog" and "A car") serving as the prompts at endpoints for interpolation. Under this setting, we apply PAID and then select the image from the interpolation sequence that achieves the highest CLIP score with respect to the compositional description.

<p align="center">
<img src="asset/composition.png">
</p>

### Image Editing

We can use [P2P](https://github.com/google/prompt-to-prompt) or [EDICT](https://github.com/salesforce/EDICT) to firstly inverse the generation process of given image, and then set the endpoint condition as the original prompt and the edting prompt, respectively, to control the editing level of images.

<p align="center">
<img src="asset/editing.png">
</p>

### Image Morphing

Using IP-Adapter, we set the two images as the condition at the endpoints of the interpolation sequence for image morphing. Notice that the text prompt can be further added to refine the generated images at the endpoints.

<p align="center">
<img src="sdxl_results/morph/1.jpg">
</p>

<p align="center">
<img src="sdxl_results/morph/2.jpg">
</p>

### Image-Control generation

Given a text prompt and an image, we can better control the scale of IP-Adapter by AID. To achieve this, we set one endpoint as only using text prompt as condition while the other endpoint using both text and image condition. This provides smoother control over the scale of IP-Adapter.

<p align="center">
<img src="sdxl_results/scale_control/1.jpg">
</p>

## 🏍️ Google Colab

Directly try PAID with [Stable Diffusion 2.1](https://colab.research.google.com/drive/1qU62G-EkcGZKSL3QRfQZQZzRuqaF94sB?usp=sharing) or [SDXL](https://colab.research.google.com/drive/1D2myS9UF241gx1otp-fy-LRakMZlROCT?usp=sharing) using Google's Free GPU!

## 🚗 Local Setup using Jupyter Notebook

1. Clone the repository and install the requirements:

``` bash
git clone https://github.com/QY-H00/attention-interpolation-diffusion.git
cd attention-interpolation-diffusion
pip install requirements.txt
```

2. Go to `play.ipynb` or `play_sdxl.ipynb` for fun!

## 🛳️ Local Setup using Gradio

1. install Gradio

``` bash
pip install gradio
```

2. Launch the Gradio interface

``` bash
gradio gradio_src/app.py
```


## 📝 Supporting Models

| Model Name            |  Link                                             |
|-----------------------|-------------------------------------------------------------|
| Stable Diffusion 1.5-512  | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)   |
| Realistic Vision V4.0 | [SG161222/Realistic_Vision_V4.0_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V4.0_noVAE) |
| Stable Diffusion 2.1-768  | [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) |
| Stable Diffusion XL-1024   | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| Animagine XL 3.1 |   [cagliostrolab/animagine-xl-3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1)|
| Realistic Vision XL V4.0 | [SG161222/RealVisXL_V5.0](https://huggingface.co/SG161222/RealVisXL_V5.0) |
| Playground v2.5 – 1024 | [playgroundai/playground-v2.5-1024px-aesthetic](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic) |
| Juggernaut XL v9 | [RunDiffusion/Juggernaut-XL-v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) |


## ✒️Citation

If you found this repository/our paper useful, please consider citing:

``` bibtex
@article{he2024aid,
  title={AID: Attention Interpolation of Text-to-Image Diffusion},
  author={He, Qiyuan and Wang, Jinghao and Liu, Ziwei and Yao, Angela},
  journal={arXiv preprint arXiv:2403.17924},
  year={2024}
}
```

## ❤️ Acknowledgement

We thank the following repositories for their great work: [diffusers](https://github.com/huggingface/diffusers), [transformers](https://github.com/huggingface/transformers), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [P2P](https://github.com/google/prompt-to-prompt) and [EDICT](https://github.com/salesforce/EDICT).
