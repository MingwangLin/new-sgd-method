import math
from torch.optim import Optimizer


class EvePlus(Optimizer):
    """
    implements EvePlus Algorithm, proposed in `IMPROVING STOCHASTIC GRADIENT DESCENT WITH FEEDBACK`
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.999), eps=1e-8,
                 k=0.1, K=10, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        k=k, K=K, weight_decay=weight_decay)
        super(EvePlus, self).__init__(params, defaults)

    def step(self, closure):
        """
        :param closure: closure returns loss. see http://pytorch.org/docs/optim.html#optimizer-step-closure
        :return: loss
        """
        loss, output = closure()
        _loss = loss.data[0]  # float

        for group in self.param_groups:

            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m_t'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['v_t'] = grad.new().resize_as_(grad).zero_()
                    # f hats, smoothly tracked objective functions
                    # \hat{f}_0 = f_0
                    state['ft_2'], state['ft_1'] = _loss, None
                    state['d'] = 1

                m_t, v_t = state['m_t'], state['v_t']
                beta1, beta2, beta3 = group['betas']
                k, K = group['k'], group['K']
                d = state['d']
                state['step'] += 1
                t = state['step']
                # initialization of \hat{f}_1
                if t == 1:
                    # \hat{f}_1 = f_1
                    state['ft_1'] = _loss
                # \hat{f_{t-1}}, \hat{f_{t-2}}
                ft_1, ft_2 = state['ft_1'], state['ft_2']
                # f(\theta_{t-1})
                f = _loss

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                m_t.mul_(beta1).add_(1 - beta1, grad)
                v_t.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                m_t_hat = m_t / (1 - beta1 ** t)
                v_t_hat = v_t / (1 - beta2 ** t)

                if  t > 1:
                    c = f / ft_2
                    r = abs(c - 1) / min(c, 1)
                    state['ft_1'], state['ft_2'] = f, ft_1
                    d_t_ema = beta3 * d + (1 - beta3) * r
                    min_d = 0.33
                    # min_d = 0.71
                    # min_d = 0.14
                    # max_d = 1
                    d_t = max(min_d, d_t_ema)
                    state['d'] = d_t
                    # state['d'] = 1
                    
#                 if  t > 391 * 30:
#                     state['d'] = 1
                # update parameters
                p.data.addcdiv_(-group['lr'] / state['d'],
                                m_t_hat,
                                v_t_hat.sqrt().add_(group['eps']))

        return loss, state['d'], output

