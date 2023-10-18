import torch # for backend
import requests # for internet access
from PIL import Image # regular python library for image processing
import numpy as np

from diffusers import StableDiffusionDepth2ImgPipeline # Hugging face pipeline

#  Creating a variable instance of the pipeline
pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True,
).to("cuda") #  Assigning to GPU

pipe.enable_sequential_cpu_offload()

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing()

def process_image(numpy_image):
    #img = img.resize((1920 // 2, 1080 // 2))
    img = Image.fromarray(numpy_image.astype('uint8'), 'RGB')
    prompt = "photorealistic characters, beautiful" #"same content, characters made realistic"
    negative_prompt = "bad anatomy, bad proportions, deformed, ugly, missing arms, missing legs, extra arms, extra legs, extra fingers, extra limbs, out of frame"
    outcome = pipe(prompt=prompt, image=img, negative_prompt=negative_prompt, strength=0.3)
    result = outcome.images[0]
    return np. array(result)
