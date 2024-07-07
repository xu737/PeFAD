# hours=3
# sleep_time=$((hours * 3600))
# echo 'sleeping...'
# sleep $sleep_time
# echo 'awake!'

print_separator() {
  printf "%s\n" "-----------------------------------------------------"
}

echo '----SWAT-----'
current_datetime=$(date "+%Y-%m-%d %H:%M")
echo "start time： $current_datetime"

# Execute the Python script with the specified parameters
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path dataset/SWaT \
  --model_id SWAT \
  --model GPT4TS \
  --data SWAT \
  --features M \
  --seq_len 100 \
  --gpt_layer 8 \
  --d_model 1280 \
  --d_ff 256 \
  --enc_in 51 \
  --c_out 51 \
  --anomaly_ratio 1 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --local_bs 32 \
  --client_nums 2 \
  --local_epoch 1 \
  --patch_len 10 \
  --patch_stride 10 \
  --mask_ratio 0.2 \
  --gpu 2 \
  --gpt True \
  --consis_loss_coef 2000 \
  --mask_factor 1.5 \
  --vae_local_epochs 10 \
  --full_tuning 0 \
  --effi_layer 1 \
  --percentile 10 \
  --connection_ratio 0.8 \

current_datetime=$(date "+%Y-%m-%d %H:%M")
echo "Finish time： $current_datetime"
