export TOKENIZERS_PARALLELISM=false
export MODEL_NAME="/root/autodl-tmp/StableAvatar/checkpoints/Wan2.1-Fun-V1.1-1.3B-InP"
export WORLD_SIZE=4
export MASTER_ADDR="localhost"
export MASTER_PORT=29500

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29500 inference.py \
  --config_path="deepspeed_config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path="/root/autodl-tmp/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt" \
  --pretrained_wav2vec_path="/root/autodl-tmp/StableAvatar/checkpoints/wav2vec2-base-960h" \
  --validation_reference_path="/root/autodl-tmp/StableAvatar/examples/case-1/reference.png" \
  --validation_driven_audio_path="/root/autodl-tmp/StableAvatar/examples/case-1/audio.wav" \
  --output_dir="/root/autodl-tmp/StableAvatar/output_infer" \
  --validation_prompts="A middle-aged woman with short light brown hair, wearing pearl earrings and a blue blazer, is speaking passionately in front of a blurred background resembling a government building. Her mouth is open mid-phrase, her expression is engaged and energetic, and the lighting is bright and even, suggesting a television interview or live broadcast. The scene gives the impression she is singing with conviction and purpose." \
  --seed=42 \
  --ulysses_degree=2 \
  --ring_degree=2 \
  --motion_frame=25 \
  --sample_steps=50 \
  --width=512 \
  --height=512 \
  --fsdp_dit \
  --overlap_window_length=10 \
  --sample_text_guide_scale=3.0 \
  --sample_audio_guide_scale=5.0
