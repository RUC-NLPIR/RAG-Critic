CUDA_VISIBLE_DEVICES=0 python3 predict_dev.py \
    --model_name_or_path output \
    --inst_file 3models_inference_merge_130_baseline.json \
    --output_path output/ \
    --dataset_name predict_1900_130 

