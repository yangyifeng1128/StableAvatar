export TOKENIZERS_PARALLELISM=false
export MODEL_NAME="/root/autodl-tmp/StableAvatar/checkpoints/Wan2.1-I2V-14B-480P"

accelerate launch --config_file accelerate_config/accelerate_config_machine_14B_multiple.yaml --machine_rank ${NODE_RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} train_14B_lora.py \
  --config_path="deepspeed_config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_wav2vec_path="/root/autodl-tmp/StableAvatar/checkpoints/wav2vec2-base-960h" \
  --validation_reference_path="/root/autodl-tmp/StableAvatar/validation/reference.png" \
  --validation_driven_audio_path="/root/autodl-tmp/StableAvatar/validation/audio.wav" \
  --train_data_rec_dir="/root/autodl-tmp/StableAvatar/talking_face_data/video_rec_path.txt" \
  --train_data_square_dir="/root/autodl-tmp/StableAvatar/talking_face_data/video_square_path.txt"  \
  --train_data_vec_dir="/root/autodl-tmp/StableAvatar/talking_face_data/video_vec_path.txt"  \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --validation_steps=500 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="/root/autodl-tmp/StableAvatar/output_14B_lora_dir_64" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --uniform_sampling \
  --motion_sub_loss \
  --use_deepspeed \
  --low_vram \
  --train_mode="i2v"