export CUDA_VISIBLE_DEVICES=0,1,2,3


Red_team_model_path=$1
Red_team_dataset=$2

OUTPUT_DIR=./model_output/red_team_dpo
current_time=$(date "+%Y%m%d%H%M%S")  
OUTPUT_DIR="${OUTPUT_DIR}_data${current_time}"
mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
True_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${True_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
SCRIPT_NAME=$(basename "$0")
DESTINATION_PATH="$OUTPUT_DIR/$SCRIPT_NAME"
cp "$0" "$DESTINATION_PATH"
num_processes=$(expr length "$CUDA_VISIBLE_DEVICES" / 2 + 1)


accelerate launch --main_process_port 29502 --config_file=script/accelerate_configs/zero2.yaml \
            --num_processes=$num_processes easyalign/algorithm_trl/dpo.py \
            --model_name_or_path $Red_team_model_path \
            --dataset_name $Red_team_dataset \
            --torch_dtype "bfloat16" \
            --attn_implementation flash_attention_2 \
            --use_peft False \
            --bf16 True \
            --load_in_8bit False \
            --load_in_4bit False \
            --do_train True \
            --num_train_epochs 1 \
            --per_device_train_batch_size 2 \
            --gradient_accumulation_steps 1 \
            --max_length 3096 \
            --max_prompt_length 2048 \
            --loss_type "sigmoid" \
            --beta 0.1 \
            --gradient_checkpointing False \
            --learning_rate 1.41e-5 \
            --optim "adamw_torch" \
            --lr_scheduler_type "cosine" \
            --warmup_ratio 0.1 \
            --logging_steps 100 \
            --save_strategy "epoch" \
            --report_to "none" \
            --output_dir $OUTPUT_DIR \



    





















