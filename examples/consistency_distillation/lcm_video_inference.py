from diffusers import DiffusionPipeline, UNet3DConditionModel, LCMScheduler
import torch
import time
from diffusers.utils import export_to_video



unet = UNet3DConditionModel.from_pretrained(
    "/home/hice1/mnigam9/scratch/cache/zeroscope_cd_distill/checkpoint-800/unet",
    torch_dtype=torch.float16,
)

pipe = DiffusionPipeline.from_pretrained(
    "/home/hice1/mnigam9/scratch/cache/zeroscope_v2_576w",
    unet=unet,
    torch_dtype=torch.float16
).to("cuda")   

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)


prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

generator = torch.manual_seed(0)
# breakpoint() 
time1 = time.time()
image = pipe(
    prompt=prompt, num_inference_steps=4
)
time2 = time.time()
print("Time taken: ", time2-time1)

# breakpoint()
frames = image.frames

print( export_to_video( frames, output_video_path="video.mp4") )