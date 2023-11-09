import torch
from torch.nn import functional as F
import torch.nn as nn
from accelerate import Accelerator
import lpips
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from consistencydecoder import ConsistencyDecoder, save_image, load_image
import argparse
import numpy as np
from PIL import Image
from datasets import load_dataset

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='Training settings')

    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lpips_scale', type=float, default=0.1,
                        help='Scale factor for LPIPS loss')
    parser.add_argument('--kl_scale', type=float, default=0.1,
                        help='Scale factor for KL divergence loss')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps for gradient accumulation')
    parser.add_argument('--dataset_name', type=str, default='dataset',
                        help='Name of the dataset to use for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--model_type', type=str, default='fp16',
                        help="Model type: 'fp32' or 'fp16'")

    return parser.parse_args()

args = parse_args()


device = "mps"

def get_consistency_decoder():
    print("Loading consistency decoder")
    decoder_consistency = ConsistencyDecoder(device=device) # Model size: 2.49 GB
    print("Consistency decoder loaded")
    return decoder_consistency

def get_sdxl_vae():
    sdxl_vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    sdxl_vae.to(device=device)

    return sdxl_vae


def get_sd_vae():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, device="cuda:0"
    )
    vae = pipe.vae

    return vae


# latent_sd_15 = pipe.vae.encode(image.half().cuda()).latent_dist.mean
# latent_sdxl = sdxl_vae.encode(image.half().cuda()).latent_dist.mean
# # decode with gan
# sample_gan = pipe.vae.decode(latent_sd_15).sample.detach()

# decode with vae

decoder_consistency = get_consistency_decoder()
# save_image(sample_consistency, "con.png")

def load_image(uri, size=None, center_crop=False):
    image = Image.open(uri)
    if center_crop:
        image = image.crop(
            (
                (image.width - min(image.width, image.height)) // 2,
                (image.height - min(image.width, image.height)) // 2,
                (image.width + min(image.width, image.height)) // 2,
                (image.height + min(image.width, image.height)) // 2,
            )
        )
    if size is not None:
        image = image.resize(size)
    image = torch.tensor(np.array(image).transpose(2, 0, 1)).unsqueeze(0).float()
    image = image / 127.5 - 1.0
    return image

# Encode with SD vae and decode with consistency decoder
#
# Accepts a PIL image
# def test_sample(image_file):
#     # pil to tensor
#     image_tensor = load_image(image_file).to(device=device)
#     latent_sd_15 = pipe.vae.encode(image_tensor.half().cuda()).latent_dist.mean

#     return image


# Defining a simple MLP (Multi-Layer Perceptron) class named 'LatentProjector'.
# This MLP is used to project the latent space of the SD VAE to the latent space of the SDXL VAE.
class LatentProjector(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=512):
        super(LatentProjector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Second layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



# Loading dataset_name from args
def get_train_dataloader(args):
    dataset_name = args.dataset_name

    dataset = load_dataset(dataset_name)["train"]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    return dataloader   


def train():
    wandb.init()

    # accelerator = Accelerator(fp16=True, device=device)
    train_dataloader = get_train_dataloader(args)

    # map args.model_type to torch dtype
    type_map ={
        "fp32": torch.float32,
        "fp16": torch.float16,
    }
    dtype = type_map[args.model_type]

    lpips_loss_fn = lpips.LPIPS(net='alex').to(device, dtype=dtype)

    latent_projetor = LatentProjector().to(device, dtype=dtype)
    latent_projetor.train()

    sdxl_vae = get_sd_vae()
    consitency_decoder = get_consistency_decoder()


    weight_dtype = torch.float16
    # logger = accelerator.get_logger()

    params_to_optimize = latent_projetor.parameters()

    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_train_epochs)

    for epoch in range(0, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # with accelerator.accumulate(latent_projetor):
            target = batch["pixel_values"].to(weight_dtype)

            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py
            posterior = sdxl_vae.encode(target).latent_dist
            z = posterior.mode()
            pred = consitency_decoder(z).sample

            kl_loss = posterior.kl().mean()
            mse_loss = F.mse_loss(pred, target, reduction="mean")
            lpips_loss = lpips_loss_fn(pred, target).mean()

            # logger.info(f'mse:{mse_loss.item()}, lpips:{lpips_loss.item()}, kl:{kl_loss.item()}')

            loss = mse_loss + args.lpips_scale * lpips_loss + args.kl_scale * kl_loss
            

            # log all losses
            wandb.log({
                "loss": loss.item(),
                "mse_loss": mse_loss.item(),
                "lpips_loss": lpips_loss.item(),
                "kl_loss": kl_loss.item(),
            })

            # Gather the losses across all processes for logging (if we use distributed training).
            # avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            # without accelerator
            avg_loss = loss.mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps


            # accelerator.backward(loss)
            # without accelerator
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

if __name__ == '__main__':
    train()
