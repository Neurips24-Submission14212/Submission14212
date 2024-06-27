import torch
import copy
import wandb
from tqdm import tqdm
from utils import dict_append
from properties_checker import compute_linear_approx, compute_smoothness, compute_difference, compute_inner_product, compute_L2_norm

# =====================================
# Standard Training (Benchmark)
# =====================================

def train_step(epoch, net, trainloader, criterion, optimizer, device,convexity_gap,L,num,denom,
               prev_loss,current_loss,train_loss,exp_avg_L_1,exp_avg_L_2,exp_avg_gap_1,exp_avg_gap_2, 
               step,train_acc,linear_actual,num_actual,sum_dif,prev_update,dif,update_product,smoothness_ratio_prev,
               smoothness_ratio_actual,current_convexity_gap,prev_grad,prev_param,current_grad_L,current_grad_linear,current_param):
    '''
    Train single epoch.
    '''
    print(f'\nTraining epoch {epoch+1}..')
    net.train()
    convexity_gap = 0
    L = 0
    num = 0
    denom = 0
    prev_loss = 0
    current_loss = 0
    train_loss = 0
    exp_avg_L_1 = 0
    exp_avg_L_2 = 0
    exp_avg_gap_1 = 0
    exp_avg_gap_2 = 0
    step = 0
    train_acc = 0
    total = 0
    correct = 0
    linear_actual = 0
    num_actual = 0
    sum_dif = 0
    prev_update = [torch.zeros_like(p) for p in net.parameters()]
    dif = [torch.zeros_like(p) for p in net.parameters()]
    update_product = 0
    smoothness_ratio_prev = 0
    smoothness_ratio_actual = 0
    current_convexity_gap = 0
    prev_grad = [torch.zeros_like(p) for p in net.parameters()]
    prev_param = [torch.zeros_like(p) for p in net.parameters()] 
    current_grad_L = [torch.zeros_like(p) for p in net.parameters()]
    current_grad_linear = [torch.zeros_like(p) for p in net.parameters()]
    current_param = [torch.zeros_like(p) for p in net.parameters()] 
    # pbar = tqdm(enumerate(trainloader))
    iterator = enumerate(trainloader)
    prev_batch = next(iterator)
    # final_loss = 0
    # print(pbar)
    for batch, (inputs, labels) in iterator:
        # load data
        # print("current iteration", batch)
        inputs, labels = inputs.to(device), labels.to(device)
        # if step >0: #updating prev_prev_param
        #     prev_loss = current_loss
        #     optimizer.save_prev_param()
        
        #compute \nabla f(w_t,x_{t-1})
        prev_batch_image = prev_batch[1][0].to(device)
        prev_batch_target = prev_batch[1][1].to(device)
        prev_batch_outputs = net(prev_batch_image) 
        prev_batch_loss = criterion(prev_batch_outputs, prev_batch_target) #f(w_t,x_{t-1})
        current_loss = prev_batch_loss.item() 
        prev_batch_loss.backward()
        i = 0
        with torch.no_grad():
            for p in net.parameters():
                current_grad_L[i].copy_(p.grad) #\nabla f(w_t,x_{t-1})
                current_param[i].copy_(p) #w_t
                i+=1
        optimizer.zero_grad()

        # forward and backward propagation
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        i = 0
        with torch.no_grad():
            for p in net.parameters():
                current_grad_linear[i].copy_(p.grad) #\nabla f(w_t,x_{t})
                i+=1
        if step >0:
            # get the inner product
            # get the smoothness constant, small L means function is relatively smooth
            dif = compute_difference(current_param, prev_param)
            sum_dif += (compute_L2_norm(dif))**2
            update_product = compute_inner_product(dif, prev_update)
            prev_update = dif
            linear_approx_prev = compute_linear_approx(current_param, current_grad_L, prev_param)
            linear_actual = compute_linear_approx(current_param, current_grad_linear, prev_param)
            current_L = compute_smoothness(current_param, current_grad_L, prev_param, prev_grad)
            L = max(L,current_L)
            # L = max(L,compute_smoothness(model, current_param, current_grad))
            denom+= current_loss - prev_loss # f(w_t,x_{t-1}) - f(w_{t-1},x_{t-1})
            smoothness_ratio_actual = (-denom + num_actual)/(1/2*sum_dif)
            smoothness_ratio_prev = (-denom + num)/(1/2*sum_dif)
            exp_avg_L_1 = 0.99*exp_avg_L_1+ (1-0.99)*current_L
            exp_avg_L_2 = 0.9999*exp_avg_L_2+ (1-0.9999)*current_L
            num_actual += compute_linear_approx(current_param, current_grad_linear, prev_param)
            # L = max(L,compute_smoothness(model, current_param, current_grad))
            # this is another quantity that we want to check: linear_approx / loss_gap. The ratio is positive is good
            num+= linear_approx_prev
            current_convexity_gap = current_loss - prev_loss - linear_approx_prev
            exp_avg_gap_1 = 0.99*exp_avg_gap_1 + (1-0.99)*current_convexity_gap
            exp_avg_gap_2 = 0.9999*exp_avg_gap_2 + (1-0.9999)*current_convexity_gap
            convexity_gap+= current_convexity_gap

        i = 0
        with torch.no_grad():
            for p in net.parameters():
                prev_grad[i].copy_(p.grad) #hold \nabla f(w_{t-1},x_{t-1}) for next iteration
                prev_param[i].copy_(p) # hold w_{t-1 } for next iteration
                i+=1
        optimizer.step()
        prev_loss = loss.item() 
        prev_batch = (batch, (inputs, labels))
        # current_loss = loss.item()
        step+=1
        # stat updates
        # inner_product += inner
        train_loss += (loss.item() - train_loss)/(batch+1)  # average train loss
        total += labels.size(0)                             # total predictions
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()        # correct predictions
        train_acc = 100*correct/total                       # average train acc
        optimizer.zero_grad()
        # prev_loss = loss.item()
        # pbar.set_description(f'epoch {epoch+1} batch {batch+1}: \
        #     train loss {train_loss:.2f}, train acc: {train_acc:.2f}, smoothness_constant:{L:.2f}, convexity_gap:{convexity_gap:.2f} ' )
        # # wandb.log({ 'smoothness_constant': L,'convexity_gap': convexity_gap })
    prev_batch_image = prev_batch[1][0].to(device)
    prev_batch_target = prev_batch[1][1].to(device)
    prev_batch_outputs = net(prev_batch_image) 
    prev_batch_loss = criterion(prev_batch_outputs, prev_batch_target) #f(w_t,x_{t-1})
    current_loss = prev_batch_loss.item() 
    prev_batch_loss.backward()
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            current_grad_L[i].copy_(p.grad) #\nabla f(w_t,x_{t-1})
            current_param[i].copy_(p) #w_t
            i+=1
    # zero grad to do the actual update
    optimizer.zero_grad()
    return prev_grad, prev_param, current_grad_L, current_grad_linear, current_param, prev_loss, current_loss, convexity_gap/(step-1), L,num/denom, num, denom,exp_avg_L_1,\
        exp_avg_L_2, exp_avg_gap_1, exp_avg_gap_2, train_acc, linear_actual, num_actual,current_convexity_gap,  smoothness_ratio_prev, smoothness_ratio_actual, update_product


def test_step(epoch, net, testloader, criterion, device):
    '''
    Test single epoch.
    '''
    print(f'\nEvaluating epoch {epoch+1}..')
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(enumerate(testloader))
    with torch.no_grad():
        for batch, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += (loss.item() - test_loss)/(batch+1)
            total += labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            # test_acc = 100*correct/total
            # pbar.set_description(f'test loss: {test_loss}, test acc: {test_acc}')

    test_acc = 100*correct/total
    print(f'test loss: {test_loss}, test acc: {test_acc}')
    return test_loss, test_acc


def train(net, trainloader, testloader, epochs, criterion, optimizer, scheduler, device,args):
    stats = {
        'args': None,
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    convexity_gap = 0
    L = 0
    num = 0
    denom = 0
    prev_loss = 0
    current_loss = 0
    train_loss = 0
    exp_avg_L_1 = 0
    exp_avg_L_2 = 0
    exp_avg_gap_1 = 0
    exp_avg_gap_2 = 0
    step = 0
    train_acc = 0
    linear_actual = 0
    num_actual = 0
    sum_dif = 0
    prev_update = [torch.zeros_like(p) for p in net.parameters()]
    dif = [torch.zeros_like(p) for p in net.parameters()]
    update_product = 0
    smoothness_ratio_prev = 0
    smoothness_ratio_actual = 0
    current_convexity_gap = 0
    prev_grad = [torch.zeros_like(p) for p in net.parameters()]
    prev_param = [torch.zeros_like(p) for p in net.parameters()] 
    current_grad_L = [torch.zeros_like(p) for p in net.parameters()]
    current_grad_linear = [torch.zeros_like(p) for p in net.parameters()]
    current_param = [torch.zeros_like(p) for p in net.parameters()] 
    if args.save:
        filename = args.path + "0.pth.tar"
        torch.save({'state_dict':optimizer.state_dict(), 'model_dict': net.state_dict() }, filename)
    for epoch in range(epochs):
        # name = 'checkpoint/'+ str(epoch+1) + ".pth.tar"
        # saved_checkpoint = torch.load(name)
        # optimizer.save_param(saved_checkpoint['state_dict'])
        # prev_loss = saved_checkpoint['current_loss']
        print("current epoch", epoch)
        prev_grad, prev_param, current_grad_L, current_grad_linear, current_param, prev_loss, current_loss, convexity_gap, smoothness,ratio,num, denom,exp_avg_L_1, \
        exp_avg_L_2, exp_avg_gap_1, exp_avg_gap_2, train_acc, linear_actual,num_actual,current_convexity_gap,  \
        smoothness_ratio_prev, smoothness_ratio_actual, update_product = train_step(epoch, net, trainloader, criterion, optimizer, device,\
                                                                                    convexity_gap,L,num,denom,prev_loss,current_loss,train_loss,exp_avg_L_1,exp_avg_L_2,exp_avg_gap_1,\
                                                                                    exp_avg_gap_2, step,train_acc,linear_actual,num_actual,sum_dif,prev_update,dif,\
                                                                                    update_product,smoothness_ratio_prev,smoothness_ratio_actual,current_convexity_gap,prev_grad,\
                                                                                    prev_param,current_grad_L,current_grad_linear,current_param )
        if args.save:
            filename = args.path + str(epoch+1) + ".pth.tar"
            torch.save({'state_dict':optimizer.state_dict(),'prev_grad':prev_grad, 
                            'prev_param': prev_param, 'current_grad_L':current_grad_L, 'current_grad_linear': current_grad_linear, 'current_param': current_param
                            , 'prev_loss':prev_loss , 'current_loss': current_loss
                          , 'model_dict': net.state_dict() }, filename)
            # save_param[epoch] = {'state_dict': copy.deepcopy(optimizer.state_dict()), 'loss':loss}
        test_loss, test_acc = test_step(epoch, net, testloader, criterion, device)

        dict_append(stats,
            test_loss=test_loss, test_acc=test_acc)
        wandb.log(
        {
            "train_loss": current_loss,
            "convexity_gap": convexity_gap,
            "smoothness": smoothness,
            "linear/loss_gap": ratio,
            "numerator" : num,
            "denominator": denom,
            'exp_avg_L_.99': exp_avg_L_1,
            'exp_avg_L_.9999': exp_avg_L_2, 
            "exp_avg_gap_.99":  exp_avg_gap_1, 
            "exp_avg_gap_.9999":  exp_avg_gap_2, 
            "train_accuracy": train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'learning rate': scheduler.get_lr()[0],
            'linear_actual': linear_actual, 
            'num_actual': num_actual, 
            'current_convexity_gap': current_convexity_gap, 
            'smoothness_ratio_prev': smoothness_ratio_prev,
            'smoothness_ratio_actual': smoothness_ratio_actual,
            'update_product': update_product,
        }
    )
        # wandb.log({
        #     'test_loss': test_loss, 'test_acc': test_acc, 'learning rate': scheduler.get_lr()[0]})
        scheduler.step()
    return stats






