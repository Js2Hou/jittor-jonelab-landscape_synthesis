import argparse
import os

import jittor as jt
from tqdm import tqdm
from omegaconf import OmegaConf

from model_jittor.dataset import InferenceDataset
from model_jittor.ldm.ddim import DDIMSampler
from model_jittor.ldm.ddpm import LatentDiffusion
from utils import to_pil_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', type=str, default='./results/')
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-i', '--id', type=int, default=0)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    args.save = os.path.join(args.save, args.name)
    os.makedirs(args.save, exist_ok=True) # FIXME: set to false
    jt.set_global_seed(args.seed)

    cfg = OmegaConf.load('./configs/inference.yaml')

    # create model
    model = LatentDiffusion(**cfg.model)

    # create ddim sampler
    # sampler = DDIMSampler(model, use_ema=False)
    
    # load dataset
    dataset = InferenceDataset(segmentation_root='/nas/landscape/test_B/labels')
    data_loader = dataset.set_attrs(batch_size=args.batch_size)
    
    # with model.ema_scope(enable=False):
    for i, (segs, names) in enumerate(tqdm(data_loader)):
        # b 29 384 512 -> b 3 96, 128
        samples = model.sample_and_decode(segs)
        predicted_image = jt.clamp((samples+1.0)/2.0, min_v=0.0, max_v=1.0)

        # samples_ddim, _ = sampler.sample(
        #     num_steps=200,
        #     condition=condition,
        #     verbose=False,
        # )
        # # b 3 96, 128 -> b 3 384 512 
        # x_samples_ddim = model.decode_first_stage(samples_ddim)

        # predicted_image = jt.clamp((x_samples_ddim+1.0)/2.0, min_v=0.0, max_v=1.0)

        for i, name in enumerate(names):
            to_pil_image(predicted_image[i]).save(f'{args.save}/{name}.jpg')
    

if __name__ == '__main__':
    jt.flags.use_cuda=True
    main()




        
    
