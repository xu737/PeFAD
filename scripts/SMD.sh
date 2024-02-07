# #!/bin/bash
print_separator() {
  printf "%s\n" "-----------------------------------------------------"
}


current_datetime=$(date "+%Y-%m-%d %H:%M")
echo "start time： $current_datetime"


echo '------SMD----------'

# Execute the Python script with the specified parameters

python -u run.py \
--task_name anomaly_detection \
--is_training 1 \
--root_path /home/data/xrh/FL/AD_FL/dataset/SMD \
--model_id SMD \
--model GPT4TS \
--data SMD \
--features M \
--seq_len 100 \
--d_model 1280 \
--d_ff 1280 \
--gpt_layer 8 \
--enc_in 38 \
--c_out 38 \
--anomaly_ratio 0.5 \
--train_epochs 4 \
--client_nums 14 \
--local_bs 32 \
--local_epoch 1 \
--patch_len 10 \
--patch_stride 10 \
--mask_ratio 0.2 \
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
--shared_size 100 \




current_datetime=$(date "+%Y-%m-%d %H:%M")
echo "Finish time： $current_datetime"