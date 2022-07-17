import argparse
import os

import jittor as jt
from omegaconf import OmegaConf
from tqdm import tqdm

from model_jittor.dataset import InferenceDataset
from model_jittor.ldm.ddim import DDIMSampler
from model_jittor.ldm.ddpm import LatentDiffusion
from utils import to_pil_image


def init_and_run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', type=str, default='./results/')
    parser.add_argument('-n', '--name', type=str, default='res67')
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args.save = os.path.join(args.save, args.name)
    os.makedirs(args.save, exist_ok=True)
    
    jt.set_global_seed(args.seed)
    cfg = OmegaConf.load('./configs/inference.yaml')

    main(args, cfg)


def main(args, cfg):
    # create model
    model = LatentDiffusion(**cfg.model)

    # create ddim sampler
    sampler = DDIMSampler(model, use_ema=False)
    
    # load dataset
    dataset = InferenceDataset(segmentation_root='/nas/landscape/test_B/labels')
    data_loader = dataset.set_attrs(batch_size=args.batch_size)
    
    for i, (segs, names) in enumerate(tqdm(data_loader)):
        # b 29 384 512 -> b 3 96, 128
        condition = model.cond_stage_model(segs)
        samples_ddim, _ = sampler.sample(
            num_steps=200,
            condition=condition,
            verbose=False,
        )
        # b 3 96, 128 -> b 3 384 512 
        # samples_ddim, _, _ = model.first_stage_model.quantize(samples_ddim)
        x_samples_ddim = model.decode_first_stage(samples_ddim)

        predicted_image = (x_samples_ddim+1.0)/2.0
        # predicted_image = jt.clamp((x_samples_ddim+1.0)/2.0, min_v=0.0, max_v=1.0)

        for i, name in enumerate(names):
            to_pil_image(predicted_image[i]).save(f'{args.save}/{name}.jpg')
    

if __name__ == '__main__':
    jt.flags.use_cuda=True
    init_and_run()




        
    
