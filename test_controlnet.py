import inspect
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers import UNet2DConditionModel, DPMSolverMultistepScheduler
from torchvision.transforms import functional as F_t

from modules.controlnet import ControlNet
from modules.controlnet.control_diffusion import ControlDiffusionPipeline
from modules.model import load_components
from modules.utils.state import load_state_dict

torch.set_grad_enabled(False)

# example_input = {
#     "sample": torch.randn(1, 4, 64, 64, device="cuda"),
#     "timestep": torch.tensor([1000], dtype=torch.int64, device="cuda"),
#     "encoder_hidden_states": torch.randn(1, 77, 768, device="cuda")
# }

MODEL = "../models/ldm/v1-5-pruned-emaonly.safetensors"
CONTROL_NET = "../models/controlnet-ssdt/control_sd15_scribble.safetensors"
VAE = "../models/vae/animevae.safetensors"

config = UNet2DConditionModel.load_config("CompVis/stable-diffusion-v1-4", subfolder="unet")
params = inspect.signature(ControlNet.__init__).parameters
params = {k: v for k, v in config.items() if k in params}

controlnet = ControlNet(**params)

state = load_state_dict(Path(CONTROL_NET))
controlnet.load_state_dict(state)
controlnet.cuda()
controlnet.enable_xformers_memory_efficient_attention()

unet, vae, encoder, _ = load_components(
    MODEL,
    vae=VAE,
    ldm_config="v1-inference.yaml",
    # clip_stop_at_layer=2
)

defaults = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear"  # Linear seems work worse
}
scheduler = DPMSolverMultistepScheduler(**defaults)

pipeline = ControlDiffusionPipeline(vae, encoder.encoder, encoder.tokenizer, unet, scheduler, None, None,
                                    requires_safety_checker=False, controlnets=[controlnet])

pipeline.to("cuda")
pipeline.set_use_memory_efficient_attention_xformers(True)


def process(image_cond_tensor, prompt, n_prompt, scale, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    final: Image.Image = pipeline(
        prompt=prompt,
        negative_prompt=n_prompt,
        num_inference_steps=10,
        guidance_scale=scale,
        num_images_per_prompt=1,
        width=512,
        height=512,
        generator=generator,
        image_conditions=image_cond_tensor,
        image_condition_cfg=False,
        # image_conditions=[image_cond_tensor],
        # image_cfg_scales=[4],
        # image_condition_cfg=True
    ).images[0]
    return [final]


def gradio_process(input_image, *args):
    return process(F_t.to_tensor(input_image["mask"][:, :, :-1]).cuda(), *args)


def create_canvas():
    return np.zeros(shape=(512, 512, 3), dtype=np.uint8) + 255


def main():
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown("## Control Stable Diffusion with Interactive Scribbles - SSDT Diffusers")
        with gr.Row():
            with gr.Column():
                create_button = gr.Button(label="Start", value='Open drawing canvas!')
                input_image = gr.Image(source='upload', type='numpy', tool='sketch')
                gr.Markdown(
                    value='Do not forget to change your brush width to make it thinner. (Gradio do not allow developers to set brush width so you need to do it manually.) '
                          'Just click on the small pencil icon in the upper right corner of the above block.')
                create_button.click(fn=create_canvas, inputs=[], outputs=[input_image])
                prompt = gr.Textbox(label="Prompt")
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=30.0, value=9.0, step=0.1)
                    seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, randomize=True)
                    n_prompt = gr.Textbox(label="Negative Prompt",
                                          value='longbody, lowres, bad anatomy, bad hands, missing fingers, pubic hair,extra digit, fewer digits, cropped, worst quality, low quality')
            with gr.Column():
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2,
                                                                                                       height='auto')
        ips = [input_image, prompt, n_prompt, scale, seed]
        run_button.click(fn=gradio_process, inputs=ips, outputs=[result_gallery])

    block.launch(server_name='127.0.0.1', server_port=1919)


if __name__ == "__main__":
    main()
