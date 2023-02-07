import inspect
from typing import List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.utils.checkpoint
import PIL
from PIL import Image
from diffusers.models import UNet2DModel, VQModel
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import PIL_INTERPOLATION, deprecate, randn_tensor
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

DEVICE = "CPU"
model_id = "CompVis/ldm-super-resolution-4x-openimages"
model_path = "./ldm-super-resolution-4x-openimages"

UNET_ONNX_PATH = Path(f"{model_path}/unet/unet.onnx")
UNET_OV_PATH = Path(f"{model_path}/unet/unet.xml")

VQVAE_ONNX_PATH = Path(f"{model_path}/vqvae/vqvae.onnx")
VQVAE_OV_PATH = Path(f"{model_path}/vqvae/vqvae.xml")

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

class OVLDMSuperResolutionPipeline(DiffusionPipeline):
    r"""
    A pipeline for image super-resolution using Latent

    This class inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vqvae ([`VQModel`]):
            Vector-quantized (VQ) VAE Model to encode and decode images to and from latent representations.
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`EulerDiscreteScheduler`],
            [`EulerAncestralDiscreteScheduler`], [`DPMSolverMultistepScheduler`], or [`PNDMScheduler`].
    """
    def __init__(
        self,
        vqvae: VQModel,
        unet: UNet2DModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()
        self.vqvae = vqvae
        self.unet = unet
        self.scheduler = scheduler
        self._unet_output = unet.output(0)
        self._vqvae_output = vqvae.output(0)
    
    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        batch_size: Optional[int] = 1,
        num_inference_steps: Optional[int] = 100,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        Args:
            image (`torch.Tensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        message = "Please use `image` instead of `init_image`."
        init_image = deprecate("init_image", "0.14.0", message, take_from=kwargs)
        image = init_image or image

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError(f"`image` has to be of type `PIL.Image.Image` or `torch.Tensor` but is {type(image)}")

        if isinstance(image, PIL.Image.Image):
            image = preprocess(image)

        height, width = image.shape[-2:]

        # in_channels should be 6: 3 for latents, 3 for low resolution image
        latents_shape = (batch_size, 3, height, width)
        latents = randn_tensor(latents_shape, generator=generator)
        # set timesteps and move to the correct device
        # self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        latents = latents.numpy()
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta
            
        for t in self.progress_bar(timesteps_tensor):
            # concat latents and low resolution image in the channel dimension.
            latents_input = np.concatenate([latents, image], axis=1)
            latents_input = self.scheduler.scale_model_input(latents_input, t)
            # predict the noise residual
            noise_pred = self.unet([latents_input, t])[self._unet_output]
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(torch.from_numpy(noise_pred), 
                                        t, 
                                        torch.from_numpy(latents))["prev_sample"].numpy()

        # decode the image latents with the VQVAE
        image = self.vqvae(latents)[self._vqvae_output]
        image = image / 2 + 0.5
        image = image.transpose(0, 2, 3, 1)

        if output_type == "pil":
            image = self.numpy_to_pil(image)
        return image

from openvino.runtime import Core
core = Core()

unet_model = core.compile_model(UNET_OV_PATH, DEVICE)
vqvae_model = core.compile_model(VQVAE_OV_PATH, DEVICE)
scheduler = LMSDiscreteScheduler.from_config("ldm-super-resolution-4x-openimages/scheduler/scheduler_config.json")

ov_pipeline = OVLDMSuperResolutionPipeline(
    unet=unet_model,
    vqvae=vqvae_model,
    scheduler=scheduler
)

# url = "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
# IMG = requests.get(url)
IMG_PATH = "ILSVRC2012_val_00000123.JPEG"
low_res_img = Image.open(IMG_PATH).convert("RGB")
low_res_img = low_res_img.resize((128, 128))

# run pipeline in inference (sample random noise and denoise)
# ov_scaled_image = ov_pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
INFER_STEP = 100
ov_scaled_image = ov_pipeline(low_res_img, num_inference_steps=INFER_STEP, eta=1)[0]

# save image
ov_scaled_image.save(IMG_PATH.split(".")[0]+f"_ov_{INFER_STEP}.jpg")
