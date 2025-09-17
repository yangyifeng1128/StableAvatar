export TOKENIZERS_PARALLELISM=false
export MODEL_NAME="/root/autodl-tmp/StableAvatar/checkpoints/Wan2.1-Fun-V1.1-1.3B-InP"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_1B_rec_vec.py \
  --config_path="deepspeed_config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_wav2vec_path="/root/autodl-tmp/StableAvatar/checkpoints/wav2vec2-base-960h" \
  --validation_reference_path="/root/autodl-tmp/StableAvatar/validation/reference.png" \
  --validation_driven_audio_path="/root/autodl-tmp/StableAvatar/validation/audio.wav" \
  --train_data_rec_dir="/root/autodl-tmp/StableAvatar/talking_face_data/video_rec_path.txt" \
  --train_data_vec_dir="/root/autodl-tmp/StableAvatar/talking_face_data/video_vec_path.txt"  \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=1000 \
  --checkpointing_steps=2000 \
  --validation_steps=500 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="/root/autodl-tmp/StableAvatar/output_1B_rec_vec_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --uniform_sampling \
  --motion_sub_loss \
  --low_vram \
  --train_mode="i2v"