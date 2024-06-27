import torch
import numpy as np
from properties_checker import compute_linear_approx, compute_smoothness,compute_L2_norm,compute_L1_norm, compute_PL,compute_inner_product, cosine_similarity
import wandb
import torchvision.models as models
import argparse

parser = argparse.ArgumentParser(description='properties checker')
parser.add_argument('--epochs', '-e', default=200, type=int, help='number of epochs')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# parser.add_argument('--name', default='experiment', type=str, help='experiment name')
parser.add_argument('--fullbatch_path', default=None, type=str, help='fullbatch path')
parser.add_argument('--opt_path', default=None, type=str, help='fullbatch path')
parser.add_argument('--path', default=None, type=str, help='path to checkpoints')
parser.add_argument('--name', default='experiment', type=str, help='experiment name')
args = parser.parse_args()


wandb.init(project='spicy_stuffs', name=args.name)
opt_checkpoint = torch.load(args.opt_path)
opt_grad =opt_checkpoint['fullbatch_grad']
opt_loss = opt_checkpoint['fullbatch_loss']
path = args.path
variance = 0
error = 0
exp_avg_num = 0
exp_avg_denom = 0
prev_num = 0
prev_denom = 0
prev_convexity_gap = 0
for e in range(args.start_epoch, args.epochs):
    print("currently in epoch", e)
    checkpoint_name = args.path+ str(e+1) + ".pth.tar"
    checkpoint = torch.load(checkpoint_name)
    # for key in checkpoint:
    #     print(key)
    # current_convexity_gap =  checkpoint['convexity_gap'] - prev_convexity_gap
    # current_num =  checkpoint['num'] - prev_num
    # current_denom = checkpoint['denom'] - prev_denom
    # prev_num = checkpoint['num']
    # prev_denom = checkpoint['denom'] 
    # prev_convexity_gap = checkpoint['convexity_gap']
    exp_avg_num = 0.99*exp_avg_num + (1-0.99)*current_num
    exp_avg_denom = 0.99*exp_avg_denom+ (1-0.99)*current_denom 
    fullbatch_name = args.fullbatch_path + str(e+1) + ".pth.tar"
    fullbatch_checkpoint = torch.load(fullbatch_name)
    current_L2_norm =  compute_L2_norm(checkpoint['current_grad'])
    current_L1_norm = compute_L1_norm(checkpoint['current_grad'])
    fullbatch_L2_norm = compute_L2_norm(fullbatch_checkpoint['fullbatch_grad'])
    fullbatch_L1_norm  = compute_L1_norm(fullbatch_checkpoint['fullbatch_grad'])
    pl_const  = compute_PL(fullbatch_checkpoint['fullbatch_grad'], fullbatch_checkpoint['fullbatch_loss'], opt_loss)
    if 'prev_grad' in checkpoint:
        inner_prod = compute_inner_product(checkpoint['current_grad'], checkpoint['prev_grad'])
        cosine = cosine_similarity(checkpoint['current_grad'], checkpoint['prev_grad'])
    dif =[]
    for i in range(len(checkpoint['current_grad'])):
        dif.append(checkpoint['current_grad'][i] - fullbatch_checkpoint['fullbatch_grad'][i])
    variance += compute_L2_norm(dif)
    wandb.log({
        'current_L2_norm': current_L2_norm,
        'current_L1_norm': current_L1_norm,
        'fullbatch_L2_norm': fullbatch_L2_norm,
        'fullbatch_L1_norm': fullbatch_L1_norm,
        'pl_const': pl_const, 
        'inner_prod_gt_g(t-1)': inner_prod, 
        'cosine_similarity_gt_g(t-1)': cosine,
        'gradient variance avg': variance/(e+1),
        'current variance': compute_L2_norm(dif),
        # 'current convexity gap': current_convexity_gap,
        # 'current_linearApprox': current_num, 
        # 'current_lossGap': current_denom,
        # 'exp_avg_num': exp_avg_num, 
        # 'exp_avg_denom': exp_avg_denom, 
        'current loss': checkpoint['current_loss'],
        # 'current fullbatch loss': fullbatch_checkpoint['fullBatch_loss'],
    })
