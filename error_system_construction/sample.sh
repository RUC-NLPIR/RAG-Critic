

#-----------------------sample-----------------------

datasets=("nq" "hotpotqa" "triviaqa" "wikiasp" "wow" "2wikimultihopqa" "asqa" "eli5" "fever") # dataset pool name
models=("Llama-3.1-8B-Instruct" "qwen-2.5-7b-instruct") # model pool name
set_names=("train_1w")

# Loop through all combinations
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for set_name in "${set_names[@]}"; do
            echo "Sampling Processing dataset: $dataset, model: $model, set: $set_name"
            
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /path/to/script/test_vllm_cot_sample.py \
                --model_name_or_path /path/to/model_zoo/${model} \
                --inst_file /path/to/dataset_pool/${dataset}/${set_name}.json \
                --output_path /path/to/output/ \
                --dataset_name ${dataset}_${set_name} \
                --debug

        done
    done
done