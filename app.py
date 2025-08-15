import torch
import psutil
import argparse
import gradio as gr
import os
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from omegaconf import OmegaConf
from wan.models.cache_utils import get_teacache_coefficients
from wan.models.wan_fantasy_transformer3d_1B import WanTransformer3DFantasyModel
from wan.models.wan_text_encoder import WanT5EncoderModel
from wan.models.wan_vae import AutoencoderKLWan
from wan.models.wan_image_encoder import CLIPModel
from wan.pipeline.wan_inference_long_pipeline import WanI2VTalkingInferenceLongPipeline
from wan.utils.fp8_optimization import replace_parameters_by_name, convert_weight_dtype_wrapper, convert_model_weight_to_float8
from wan.utils.utils import get_image_to_video_latent, save_videos_grid
import numpy as np
import librosa
import datetime
import random
import math
import subprocess


parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
args = parser.parse_args()


if torch.cuda.is_available():
    device = "cuda" 
    if torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


model_path = "checkpoints"
pretrained_model_name_or_path = f"{model_path}/Wan2.1-Fun-V1.1-1.3B-InP"
pretrained_wav2vec_path = f"{model_path}/wav2vec2-base-960h"
transformer_path = f"{model_path}/StableAvatar-1.3B/transformer3d-square.pt"
config = OmegaConf.load("deepspeed_config/wan2.1/wan_civitai.yaml")
sampler_name = "Flow"
clip_sample_n_frames = 81
tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')), )
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=dtype,
)
text_encoder = text_encoder.eval()
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
)
wav2vec_processor = Wav2Vec2Processor.from_pretrained(pretrained_wav2vec_path)
wav2vec = Wav2Vec2Model.from_pretrained(pretrained_wav2vec_path).to("cpu")
clip_image_encoder = CLIPModel.from_pretrained(os.path.join(pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')), )
clip_image_encoder = clip_image_encoder.eval()
transformer3d = WanTransformer3DFantasyModel.from_pretrained(
    os.path.join(pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=False,
    torch_dtype=dtype,
)
if transformer_path is not None:
    state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = transformer3d.load_state_dict(state_dict, strict=False)
Choosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
}[sampler_name]
scheduler = Choosen_Scheduler(
    **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)
pipeline = WanI2VTalkingInferenceLongPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    transformer=transformer3d,
    clip_image_encoder=clip_image_encoder,
    scheduler=scheduler,
    wav2vec_processor=wav2vec_processor,
    wav2vec=wav2vec,
)


def generate(
    GPU_memory_mode,
    teacache_threshold,
    num_skip_start_steps,
    image_path,
    audio_path,
    prompt,
    negative_prompt,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    text_guide_scale,
    audio_guide_scale,
    motion_frame,
    fps,
    overlap_window_length,
    seed_param,
):
    global pipeline, transformer3d
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if seed_param<0:
        seed = random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = seed_param

    if GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer3d, ["modulation", ], device=device)
        transformer3d.freqs = transformer3d.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer3d, exclude_module_name=["modulation", ])
        convert_weight_dtype_wrapper(transformer3d, dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)
        
    if teacache_threshold > 0:
        coefficients = get_teacache_coefficients(pretrained_model_name_or_path)
        pipeline.transformer.enable_teacache(
            coefficients,
            num_inference_steps,
            teacache_threshold,
            num_skip_start_steps=num_skip_start_steps,
            #offload=args.teacache_offload
        )

    with torch.no_grad():
        video_length = int((clip_sample_n_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if clip_sample_n_frames != 1 else 1
        input_video, input_video_mask, clip_image = get_image_to_video_latent(image_path, None, video_length=video_length, sample_size=[height, width])
        sr = 16000
        vocal_input, sample_rate = librosa.load(audio_path, sr=sr)
        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            num_inference_steps=num_inference_steps,
            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
            text_guide_scale=text_guide_scale,
            audio_guide_scale=audio_guide_scale,
            vocal_input_values=vocal_input,
            motion_frame=motion_frame,
            fps=fps,
            sr=sr,
            cond_file_path=image_path,
            overlap_window_length=overlap_window_length,
            seed=seed,
            overlapping_weight_scheme="uniform",
        ).videos
        os.makedirs("outputs", exist_ok=True)
        video_path = os.path.join("outputs", f"{timestamp}.mp4")
        save_videos_grid(sample, video_path, fps=fps)
        output_video_with_audio = os.path.join("outputs", f"{timestamp}_audio.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "quiet", "-i", video_path, "-i", audio_path, 
            "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", 
            output_video_with_audio
        ], check=True)
        
    return output_video_with_audio, seed, f"已生成outputs/{timestamp}.mp4"


def exchange_width_height(width, height):
    return height, width, "✅ 宽高交换完毕"


def adjust_width_height(image):
    image = load_image(image)
    width, height = image.size
    original_area = width * height
    default_area = 512*512
    ratio = math.sqrt(original_area / default_area)
    width = width / ratio // 16 * 16
    height = height / ratio // 16 * 16
    return int(width), int(height), "✅ 根据图片调整宽高"


with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">StableAvatar</h2>
            </div>
            """)
    with gr.Accordion("模型设置", open=False):
        with gr.Row():
            GPU_memory_mode = gr.Dropdown(
                label = "显存模式", 
                info = "Normal占用25G显存，model_cpu_offload占用13G显存", 
                choices = ["Normal", "model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"], 
                value = "model_cpu_offload"
            )
            teacache_threshold = gr.Slider(label="teacache threshold", info = "推荐参数0.1，0为禁用teacache加速", minimum=0, maximum=1, step=0.01, value=0)
            num_skip_start_steps = gr.Slider(label="跳过开始步数", info = "推荐参数5", minimum=0, maximum=100, step=1, value=5)
    with gr.TabItem("StableAvatar"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image_path = gr.Image(label="上传图片", type="filepath", height=280)
                    audio_path = gr.Audio(label="上传音频", type="filepath")
                prompt = gr.Textbox(label="提示词", value="")
                negative_prompt = gr.Textbox(label="负面提示词", value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
                generate_button = gr.Button("🎬 开始生成", variant='primary')
                with gr.Accordion("参数设置", open=True):
                    with gr.Row():
                        width = gr.Slider(label="宽度", minimum=256, maximum=2048, step=16, value=512)
                        height = gr.Slider(label="高度", minimum=256, maximum=2048, step=16, value=512)
                    with gr.Row():
                        exchange_button = gr.Button("🔄 交换宽高")
                        adjust_button = gr.Button("根据图片调整宽高")
                    with gr.Row():
                        guidance_scale = gr.Slider(label="guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=6.0)
                        num_inference_steps = gr.Slider(label="采样步数（推荐50步）", minimum=1, maximum=100, step=1, value=50)
                    with gr.Row():
                        text_guide_scale = gr.Slider(label="text guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=3.0)
                        audio_guide_scale = gr.Slider(label="audio guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=5.0)
                    with gr.Row():
                        motion_frame = gr.Slider(label="motion frame", minimum=1, maximum=50, step=1, value=25)
                        fps = gr.Slider(label="帧率", minimum=1, maximum=60, step=1, value=25)
                    with gr.Row():
                        overlap_window_length = gr.Slider(label="overlap window length", minimum=1, maximum=20, step=1, value=5)
                        seed_param = gr.Number(label="种子，请输入正整数，-1为随机", value=-1)
            with gr.Column():
                info = gr.Textbox(label="提示信息", interactive=False)
                video_output = gr.Video(label="生成结果", interactive=False)
                seed_output = gr.Textbox(label="种子")

    gr.on(
        triggers=[generate_button.click, prompt.submit, negative_prompt.submit],
        fn = generate,
        inputs = [
            GPU_memory_mode,
            teacache_threshold,
            num_skip_start_steps,
            image_path,
            audio_path,
            prompt,
            negative_prompt,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            text_guide_scale,
            audio_guide_scale,
            motion_frame,
            fps,
            overlap_window_length,
            seed_param,
        ],
        outputs = [video_output, seed_output, info]
    )
    exchange_button.click(
        fn=exchange_width_height, 
        inputs=[width, height], 
        outputs=[width, height, info]
    )
    adjust_button.click(
        fn=adjust_width_height, 
        inputs=[image_path], 
        outputs=[width, height, info]
    )


if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )