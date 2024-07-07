print_separator() {
  printf "%s\n" "-----------------------------------------------------"
}


current_datetime=$(date "+%Y-%m-%d %H:%M")
echo "start time： $current_datetime"

echo 'MSL'


# Execute the Python script with the specified parameters
python -u run.py \
--task_name anomaly_detection \
--is_training 1 \
--root_path dataset/MSL \
--model_id MSL \
--model GPT4TS \
--data MSL \
--features M \
--seq_len 100 \
--gpt_layer 8 \
--d_model 1280 \
--d_ff 16 \
--enc_in 55 \
--c_out 55 \
--anomaly_ratio 2 \
--learning_rate 0.00001 \
--train_epochs 2\
--client_nums 8 \
--local_bs 32 \
--local_epoch 1 \
--patch_len 10 \
--patch_stride 10 \
--mask_ratio 0.3 \
--gpu 2 \
--gpt True \
--consis_loss_coef 100000  \
--mask_factor 2 \
--vae_local_epochs 20 \
--full_tuning 0\
--effi_layer 1 \
--percentile 10 \
--connection_ratio 0.6 \
--continue_training 0 \



current_datetime=$(date "+%Y-%m-%d %H:%M")
echo "finish time： $current_datetime"
