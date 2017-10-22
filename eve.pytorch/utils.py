import numpy as np

# linearly increase learning rate for purpose of determine the converged learning rate.
## old
def lr_increased_linearly(epoch_index, batch_index):
    lr_max = 0.05
    batch_num = 391
    epoch_num = 8
    lr = 0.05 * (epoch_index / epoch_num) * (batch_index+1) / (batch_num)
    return lr

## old
def lr_decreased_linearly(epoch_index, batch_index):
    lr_initial = 0.001
    decay_rate = 0.001
    epoch_num = 8
    lr = 0.001 / ( 1 + (batch_index+1) * epoch_index * decay_rate)
    return lr

def lr_down_linearly(optimizer, epoch_index, batch_index):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_initial = 0.001
    decay_rate = 0.0001
    lr = lr_initial / ( 1 + (batch_index+1) * epoch_index * decay_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def lr_down_linearly_v3(optimizer, epoch_index, batch_index):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_initial = 0.001
    decay_rate = 0.0001
    lr = lr_initial / ( 1 + (batch_index + 1 + 391 * (epoch_index - 1)) * decay_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def lr_down_linearly_v2(optimizer, epoch_index, batch_index):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_initial = 0.001
    epochs = 50
    iters_per_epoch = 391
    lr = lr_initial * (1 - (batch_index + 1 + 391 * (epoch_index - 1)) / (iters_per_epoch * epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer 

def lr_down_square(optimizer, epoch_index, batch_index):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_initial = 0.001
    epochs = 50
    iters_per_epoch = 391
    lr = lr_initial * np.square((1 - (batch_index + 1 + 391 * (epoch_index - 1)) / (iters_per_epoch * epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer 
    
def lr_down_sqrt(optimizer, epoch_index, batch_index):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_initial = 0.001
    epochs = 50
    iters_per_epoch = 391
    lr = lr_initial * np.sqrt((1 - (batch_index + 1 + 391 * (epoch_index - 1)) / (iters_per_epoch * epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer 

def lr_down_cyclically_e4(optimizer, lr_initial, epoch_index, batch_index):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_base = 0
    ites_per_epoch = 391
    epochs_per_cycle = 4
    decay_step = (lr_initial - lr_base) / (ites_per_epoch * epochs_per_cycle)

    if epoch_index in (27, 31, 35, 39, 43, 47, 51) and batch_index == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_initial
    else:
        for param_group in optimizer.param_groups:
            lr_before = param_group['lr'] 
            lr_next = lr_before - decay_step
            param_group['lr'] = lr_next
    return optimizer

def lr_down_cyclically_e8(optimizer, lr_initial, epoch_index, batch_index):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_base = 0
    ites_per_epoch = 391
    epochs_per_cycle = 8
    decay_step = (lr_initial - lr_base) / (ites_per_epoch * epochs_per_cycle)

    if epoch_index in (27, 35, 43) and batch_index == 0:

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_initial
    else:
        for param_group in optimizer.param_groups:
            lr_before = param_group['lr'] 
            lr_next = lr_before - decay_step
            param_group['lr'] = lr_next
    return optimizer

def lr_down_cyclically_a4(optimizer, lr_initial, epoch_index, batch_index):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_base = 0
    ites_per_epoch = 391
    epochs_per_cycle = 4
    # decay_step = lr_initial / (ites_per_epoch * epochs_per_cycle)

    decay_step = (lr_initial - lr_base) / (ites_per_epoch * epochs_per_cycle)


    if (epoch_index - 1) % 4 == 0 and batch_index == 0:

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_initial
    else:
        for param_group in optimizer.param_groups:
            lr_before = param_group['lr'] 
            lr_next = lr_before - decay_step
            param_group['lr'] = lr_next
    return optimizer

def lr_down_cyclically_a8(optimizer, lr_initial, epoch_index, batch_index):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_base = 0
    ites_per_epoch = 391
    epochs_per_cycle = 4
    decay_step = (lr_initial - lr_base) / (ites_per_epoch * epochs_per_cycle)
    if (epoch_index - 1) % 8 == 0 and batch_index == 0:

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_initial
    else:
        for param_group in optimizer.param_groups:
            lr_before = param_group['lr'] 
            lr_next = lr_before - decay_step
            param_group['lr'] = lr_next
    return optimizer

# range test
def lr_up_linearly(optimizer, epoch_index, batch_index):
    lr_max = 0.05
    batch_num = 391
    epoch_num = 4
    lr_step = lr_max / (batch_num * epoch_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] + lr_step

    return optimizer

def lr_down_fixed_step(optimizer, epoch_index, batch_index):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch_index <= 15:
        lr = 0.001
    elif epoch_index > 15 and epoch_index <= 30:
        lr = 0.0001
    elif epoch_index > 30:
        lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer