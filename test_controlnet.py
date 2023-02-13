import inspect
from pathlib import Path
from types import MethodType

import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers import UNet2DConditionModel, StableDiffusionPipeline, DPMSolverMultistepScheduler
from torchvision.transforms import functional as F_t

from modules.controlnet import ControlNet, ControlledUNet
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
    # clip_stop_at_layer=2
)

defaults = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear"  # Linear seems work worse
}
scheduler = DPMSolverMultistepScheduler(**defaults)

vae.cuda()

image_cond_tensor = None


def hook_forward(*args, **kwargs):
    control_kwargs = kwargs.copy()
    control_kwargs["image_condition"] = image_cond_tensor
    control = controlnet(*args[1:], **control_kwargs)

    kwargs["down_connect_hidden_states"] = control.down_connect_hidden_states
    kwargs["mid_connect_hidden_state"] = control.mid_connect_hidden_state

    return ControlledUNet.forward(*args, **kwargs)


unet.forward = MethodType(hook_forward, unet)

pipeline = StableDiffusionPipeline(vae, encoder.encoder, encoder.tokenizer, unet, scheduler, None, None, False)

pipeline.to("cuda")
pipeline.set_use_memory_efficient_attention_xformers(True)


def process(input_image, prompt, n_prompt, scale, seed):
    global image_cond_tensor
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    image_cond_tensor = F_t.to_tensor(input_image["mask"][:, :, :-1]).cuda()
    final: Image.Image = pipeline(
        prompt=prompt,
        negative_prompt=n_prompt,
        num_inference_steps=10,
        guidance_scale=scale,
        num_images_per_prompt=1,
        width=512,
        height=512,
        generator=generator
    ).images[0]
    return [final]


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
        run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

    block.launch(server_name='127.0.0.1', server_port=1919)


if __name__ == "__main__":
    main()
