import torch
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from diffusers import LDMSuperResolutionPipeline



device = "cpu"
model_id = "CompVis/ldm-super-resolution-4x-openimages"
model_path = "./ldm-super-resolution-4x-openimages"

# load model and scheduler
pipeline = LDMSuperResolutionPipeline.from_pretrained(model_path)
pipeline = pipeline.to(device)

UNET_ONNX_PATH = Path(f"{model_path}/unet/unet.onnx")
UNET_OV_PATH = Path(f"{model_path}/unet/unet.xml")

def convert_unet_onnx(pipe: LDMSuperResolutionPipeline, onnx_path: Path):
    latents_shape = (1, 6, 128, 128)
    latents = torch.randn(latents_shape).to(device)
    t = torch.from_numpy(np.array(1, dtype=float)).to(device)
    with torch.no_grad():
        torch.onnx.export(
            pipe.unet, 
            (latents, t), str(onnx_path),
            input_names=['sample', 'timestep'],
            output_names=['out_sample']
        )
    print('Unet successfully converted to ONNX')

if not UNET_ONNX_PATH.exists():
    convert_unet_onnx(pipeline, UNET_ONNX_PATH)
    UNET_MO_CMD = f"mo -m {UNET_ONNX_PATH} -o {model_path}/unet/ "
    os.system(UNET_MO_CMD)


VQVAE_ONNX_PATH = Path(f"{model_path}/vqvae/vqvae.onnx")
VQVAE_OV_PATH = Path(f"{model_path}/vqvae/vqvae.xml")

def convert_vqvae_onnx(pipe: LDMSuperResolutionPipeline, onnx_path: Path):  

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vqvae):
            super().__init__()
            self.vqvae = vqvae

        def forward(self, latents):
            return self.vqvae.decode(latents)

    vqvae_decoder = VAEDecoderWrapper(pipe.vqvae)
    latents_shape = (1, 6, 128, 128)
    latents = torch.randn(latents_shape).to(device)
    t = torch.from_numpy(np.array(1, dtype=float))

    output_latents = pipe.unet(latents, t)[0]
    print(output_latents.shape)
    latents_uncond = output_latents[0].unsqueeze(0)
    latents_new = latents_uncond 

    vqvae_decoder.eval()
    with torch.no_grad():
        torch.onnx.export(
            vqvae_decoder, 
            latents_new, str(onnx_path),
            input_names=['latents'],
            output_names=['out_sample'],
            opset_version=13,
            # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        )
    print('VQVAE successfully converted to ONNX')


if not VQVAE_ONNX_PATH.exists():
    convert_vqvae_onnx(pipeline, VQVAE_ONNX_PATH)
    VQVAE_MO_CMD = f"mo -m {VQVAE_ONNX_PATH} -o {model_path}/vqvae/ "
    os.system(VQVAE_MO_CMD)