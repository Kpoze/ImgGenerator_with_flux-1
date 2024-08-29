import torch
import matplotlib.pyplot as plt
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("your path", torch_dtype=torch.bfloat16)
#pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()

prompt = "a watermelon walking on the moon"
img = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    #height=768,
    #width=1360,
    output_type="pil",
    num_inference_steps=4,
    max_sequence_length=256,
    #generator=torch.Generator('cpu').manual_seed(0)
    generator=torch.Generator('cuda').manual_seed(0)
).images[0]

plt.imshow(img)
plt.show()
img.save("image.png")
