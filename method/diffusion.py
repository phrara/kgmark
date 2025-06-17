import sched
import torch
from tqdm import *
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, DDIMScheduler


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_id = '/home/phr/projects/models/hf/sd-dreambooth-library/mr-potato-head'
# model_id = '../models/hf/google/ddpm-cifar10-32'


# 加载 Inversion Scheduler
inverse_sche = DDIMInverseScheduler.from_pretrained(
    model_id, 
    subfolder='scheduler')

# 加载 Pipeline
sample_pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    safety_checker=None,
    torch_dtype=torch.float).to(device)

inverse_pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=inverse_sche,
    safety_checker=None,
    torch_dtype=torch.float).to(device)



class SimpleDDIMScheduler:
    def __init__(self, batch_size, num_inference_steps=50, guidance_scale=1.):
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        text_inputs = sample_pipe.tokenizer(
            ['']*batch_size,
            padding="max_length",
            max_length=sample_pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        self.encoder_hidden_states = sample_pipe.text_encoder(text_inputs.input_ids.to("cuda"))[0]

    def inverse(self, z):
        inverse_latent, _ = inverse_pipe(
            prompt=['']*self.batch_size, 
            negative_prompt=['']*self.batch_size,
            num_inference_steps=self.num_inference_steps, 
            guidance_scale=self.guidance_scale,
            output_type='latent',
            return_dict=False,
            latents=z
        )
        return inverse_latent
    
    def inverse_all_step(self, z, step=None):
        inverse_pipe.scheduler.set_timesteps(self.num_inference_steps if step is None else step)

        inversed_latents = []
        for i, t in tqdm(enumerate(inverse_pipe.scheduler.timesteps)):
            # 执行逆采样的一步
            with torch.no_grad():
                noise_pred = inverse_pipe.unet(
                    z, t, encoder_hidden_states=self.encoder_hidden_states).sample
            z = inverse_pipe.scheduler.step(noise_pred, t, z).prev_sample

            inversed_latents.append(z)
        
        return torch.stack(inversed_latents)

    def sample(self, inverse_latent):
        new_latent, _ = sample_pipe(
            prompt=['']*self.batch_size, 
            negative_prompt=['']*self.batch_size,
            num_inference_steps=self.num_inference_steps, 
            guidance_scale=self.guidance_scale,
            output_type='latent',
            return_dict=False,
            latents=inverse_latent
        )
        return new_latent
    
    
    def sample_all_step(self, z, step=None):
        sample_pipe.scheduler.set_timesteps(self.num_inference_steps if step is None else step)
        
        latents = []
        for i, t in tqdm(enumerate(sample_pipe.scheduler.timesteps)):
            # 执行采样的一步
            with torch.no_grad():
                noise_pred = sample_pipe.unet(
                    z, t, encoder_hidden_states=self.encoder_hidden_states).sample
            z = sample_pipe.scheduler.step(noise_pred, t, z).prev_sample

            latents.append(z)
        
        return torch.stack(latents)