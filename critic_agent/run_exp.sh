save_note='experiment 1'
model_name='qwen-2.5-7b-instruct'
model_path='Qwen/qwen-2.5-7B-Instruct'

python plan_agent.py \
    --model_name $model_name  \
    --plan_model_name 'qwen2.5-72B-instruct' \
    --plan_model_path 'Qwen/Qwen2.5-72B-Instruct' \
    --sample_num 100 \
    --dataset_name 'nq' \
    --split 'test' \
    --retrieval_data_dir 'retrieval_results/' \
    --previous_data_dir 'previous_answer_data/' \
    --error_data_dir 'error_data/' \
    --config_path 'myconfig.yaml' \
    --save_dir 'results/plan/' \
    --save_note $save_note \
    && \

python execute_agent.py \
    --model_name $model_name  \
    --model_path $model_path \
    --plan_model_name 'qwen2.5-72B-instruct' \
    --sample_num 100 \
    --dataset_name 'nq' \
    --split 'test' \
    --plan_data_dir 'results/plan/' \
    --retrieval_data_dir 'retrieval_results/' \
    --previous_data_dir 'previous_answer_data/' \
    --error_data_dir 'error_data/' \
    --retrieval_method 'e5' \
    --retrieval_model_path 'intfloat/e5-base-v2' \
    --config_path 'myconfig.yaml' \
    --save_dir 'results/execute/' \
    --save_note $save_note \
    && \

