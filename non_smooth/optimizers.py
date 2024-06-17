import torch


class O2NCGD(torch.optim.Optimizer):
    """Integrated implemention of O2NC with OGD-MD.

    Updates x_t = x_{t-1} + s_t*Delta_t,
            Delta_{t+1} = (Delta_t - eta_t * g_t) * [beta / (1 + eta_t*mu)]
    """

    def __init__(self, params, lr, beta=1.0, mu=0.0, random_scaling=False):
        defaults = dict(lr=lr, beta=beta, mu=mu)
        super(O2NCGD, self).__init__(params, defaults)
        self.random_scaling = random_scaling
        self.scalar = 1.0
        # Initialize states.
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['Delta'] = torch.zeros_like(p.data)
        

    def step(self, closure = None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Sample a global random scalar first.
        self.scalar = torch.distributions.Exponential(rate=1).sample() if self.random_scaling else 1.0

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            mu = group['mu']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                # Update OGD-MD.
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Does not support sparse gradients')
                
                state = self.state[p]
                state['step'] += 1
                state['Delta'] = (state['Delta'] - lr*grad) * beta / (1+lr*mu)

                # Update O2NC.
                p.data += self.scalar * state['Delta']


class O2NCAdamW(torch.optim.Optimizer):
    """Integrated implemention of Randomized AdamW."""

    def __init__(self, params, lr, b1=0.9, b2=0.999, wd=0.0, eps=1e-8, random_scaling=False):
        defaults = dict(lr=lr, b1=b1, b2=b2, wd=wd, eps=eps)
        super(O2NCAdamW, self).__init__(params, defaults)
        self.random_scaling = random_scaling
        self.scalar = 1.0
        # Initialize states.
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['mu'] = torch.zeros_like(p.data)
                state['nu'] = torch.zeros_like(p.data)
                state['Delta'] = torch.zeros_like(p.data)
        

    def step(self, closure = None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Sample a global random scalar first.
        self.scalar = torch.distributions.Exponential(rate=1).sample() if self.random_scaling else 1.0

        for group in self.param_groups:
            lr = group['lr']
            b1 = group['b1']
            b2 = group['b2']
            wd = group['wd']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Does not support sparse gradients')
                
                state = self.state[p]
                state['step'] += 1
                state['mu'] = b1*state['mu'] + (1-b1)*grad
                state['nu'] = b2*state['nu'] + (1-b2)*grad**2
                mu_hat = state['mu']/(1-b1**state['step'])
                nu_hat = state['nu']/(1-b2**state['step'])
                state['Delta'] = -lr * (mu_hat / (eps + torch.sqrt(nu_hat)) + wd * p.data)

                p.data += self.scalar * state['Delta']
