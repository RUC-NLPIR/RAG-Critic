datasets=("nq" "hotpotqa" "triviaqa" "wikiasp" "wow" "2wikimultihopqa" "asqa" "eli5" "fever")
models=("qwen-2.5-7b-instruct" "Llama-3.1-8B-Instruct")
critic_model="qwen-2.5-7b-instruct"
set_names=("train_1w")

# Loop through all combinations
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for set_name in "${set_names[@]}"; do
            echo "Tagging Processing dataset: $dataset, test_model_response: $model, Critic_model: $critic_model, set: $set_name"
            
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /path/to/script/error_type_tagger.py \
                --model_name_or_path /path/to/model_zoo/${critic_model} \
                --inst_file /path/to/error_critic_results/critic_${model}_${dataset}_${set_name}.json \
                --output_path /path/to/error_tag_results/ \
                --dataset_name ${dataset}_${set_name} \
                --debug
        done
    done
done