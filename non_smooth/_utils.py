import torch


def tree_subtract(tree1, tree2):
    """Returns tree1-tree2. Both named_parameters."""
    return {name: param.add(-tree2[name]) for name, param in tree1.items()}


def tree_inner(tree1, tree2):
    """Returns inner product of leaves of tree1 and tree2."""
    inner = 0
    for name, param in tree1.items():
        inner += torch.dot(
            torch.flatten(param),
            torch.flatten(tree2[name])
        ).item()
    return inner


def tree_norm_l2(tree):
    """Returns the l2 norm of flattened leaves."""
    norm_sq = 0
    for _, param in tree.items():
        norm_sq += torch.sum(param**2).item()
    return norm_sq**0.5


def get_opt_state(state_name, optimizer, model):
    """Returns a tree of optimizer states."""
    output = {}
    for name, param in model.named_parameters():
        state = optimizer.state[param]
        output.update({
            name: state.get(state_name, None)
        })
    return output


def compute_prev_loss(model, prev_params, batch):
    """Computes loss of model at prev_params. prev_params is a {name: param} dict."""
    current_params = {name: param.data.clone() for name, param in model.named_parameters()}
    try:
        # Replace model parameters with cloned parameters
        for name, param in model.named_parameters():
            param.data.copy_(prev_params[name])
        with torch.no_grad():
            loss = model(**batch).loss
    finally:
        # Restore original parameters
        for name, param in model.named_parameters():
            param.data.copy_(current_params[name])
    return loss
