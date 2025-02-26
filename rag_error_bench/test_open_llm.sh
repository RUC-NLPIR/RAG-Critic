

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /share/project/dgt/rag_error_bench/predict_dev.py \
    --model_name_or_path /share/project/huitingfeng/model_zoo/qwen-2.5-7b-instruct \
    --inst_file /share/project/dgt/rag_error_bench/test_data/baseline_test.json \
    --output_path /share/project/dgt/rag_error_bench/error_sample_results/ \
    --dataset_name predict_1900
