CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --stage sft \
    --model_name_or_path 'models/llama2-13b-chat' \
    --checkpoint_dir 'lora/llama2-13b-chat-v1' \
    --model_name 'llama' \
    --template 'llama2' \
    --do_predict \
    --input_file 'data/input.json' \
    --output_file 'results/llama2-13b-chat-v1_output.json' \
    --finetuning_type lora \
    --output_dir 'lora/test' \
    --predict_with_generate \
    --max_source_length 512 \
    --bf16 \
    --max_new_tokens 300