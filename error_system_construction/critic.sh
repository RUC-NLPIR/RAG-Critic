
datasets=("nq" "hotpotqa" "triviaqa" "wikiasp" "wow" "2wikimultihopqa" "asqa" "eli5" "fever")
models=("Llama-3.1-8B-Instruct" "qwen-2.5-7b-instruct")
critic_model="qwen-2.5-7b-instruct"
set_names=("train_1w")

# Loop through all combinations
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for set_name in "${set_names[@]}"; do
            echo "Critic Processing dataset: $dataset, test_model_response: $model, Critic_model: $critic_model, set: $set_name"
            
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /path/to/script/test_critic.py \
                --model_name_or_path /path/to/model_zoo/${critic_model} \
                --inst_file /path/to/error_sampling_results/responses_${model}_${dataset}_${set_name}.json \
                --output_path /path/to/error_critic_results/ \
                --dataset_name ${dataset}_${set_name}
        done
    done
done