
print_separator() {
  printf "%s\n" "-----------------------------------------------------"
}

current_datetime=$(date "+%Y-%m-%d %H:%M")
echo "start time： $current_datetime"


# echo 'FF'
echo '----PSM-----'

python -u run.py \
--task_name anomaly_detection \
--is_training 1 \
--root_path ./dataset/PSM \
--model_id PSM \
--model GPT4TS \
--data PSM \
--features M \
--seq_len 100 \
--gpt_layer 8 \
--d_model 1280 \
--d_ff 1280 \
--enc_in 25 \
--c_out 25 \
--anomaly_ratio 1 \
--learning_rate 0.0001 \
--train_epochs 5 \
--local_bs 32 \
--client_nums 4 \
--local_epoch 1 \
--patch_len 10 \
--patch_stride 10 \
--mask_ratio 0.4 \
--gpu 1 \
--gpt True \
--consis_loss_coef 10 \
--mask_factor 1.5 \
--vae_local_epochs 10 \
--full_tuning 0 \
--effi_layer 3 \
--percentile 10 \
--connection_ratio 0.8 \
--continue_training 0 \
        


current_datetime=$(date "+%Y-%m-%d %H:%M")
echo "finish time： $current_datetime"
