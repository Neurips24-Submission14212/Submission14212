import torch 
import numpy as np
import math

def compute_linear_approx(current_param, current_grad, prev_param):
    #this function compute the following inner product: <\nabla f(x_t), x_t - x_{t-1}>
    # It takes 2 arguments:
    # 1. current_param: the param at iteration t (a list contains all param)
    # 2. current_grad: the gradient evaluated at current_param
    # 2. prev_param: the model parameters in the previous iteration (x_{t-1})
    linear_approx = 0
    # i = 0
    for i in range(len(current_param)):
        linear_approx+= torch.dot(torch.flatten(current_grad[i]), torch.flatten(current_param[i].add(-prev_param[i]))).item()
    return linear_approx

def compute_smoothness(current_param, current_grad, prev_param, prev_grad):
    sum_num = 0
    sum_denom = 0
    for i in range(len(current_param)):
        sum_num += torch.norm(current_grad[i] - prev_grad[i])
        sum_denom += torch.norm(current_param[i] - prev_param[i])
    return sum_num/sum_denom


def compute_L2_norm(grad):
    total_norm = 0
    for i in range(len(grad)):
        grad_norm = grad[i].data.norm(2)
        total_norm += grad_norm.item()**2
    return total_norm**(1./2)

def compute_L1_norm(grad):
    total_norm = 0
    for i in range(len(grad)):
        grad_norm = grad[i].data.norm(1)
        total_norm += grad_norm.item()
    return total_norm
    
    
def compute_PL(current_grad,  current_loss, opt_loss):
    grad_norm_squared = compute_L2_norm(current_grad)**2
    return (1/2)*grad_norm_squared /(current_loss - opt_loss)

def compute_inner_product(first_param, second_param):
    inner_prod = 0
    for i in range(len(first_param)):
        inner_prod+= torch.dot(torch.flatten(first_param[i]),torch.flatten(second_param[i])).item()
    return inner_prod
    
def cosine_similarity(first_param, second_param):
    inner_product = compute_inner_product(first_param, second_param)
    first_norm = compute_L2_norm(first_param)
    second_norm = compute_L2_norm(second_param)
    return inner_product/(first_norm*second_norm)

def compute_angle(current_param, current_grad, prev_param):
    linear_approx = compute_linear_approx(current_param, current_grad, prev_param)
    dif =[]
    for i in range(len(current_param)):
        dif.append(current_param[i] - prev_param[i])
    return linear_approx/(compute_L2_norm(current_grad)*compute_L2_norm(dif))

def compute_difference(current_param, prev_param):
    dif =[]
    for i in range(len(current_param)):
        dif.append(current_param[i] - prev_param[i])       
    return dif 
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
