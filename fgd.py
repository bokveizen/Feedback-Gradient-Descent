import torch
from torch.optim.optimizer import Optimizer, required

class SGDStiefel(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False, stiefel=0., feedback=3):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        stiefel=stiefel, feedback=feedback)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDStiefel, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDStiefel, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            stiefel = group['stiefel']
            baselr = group['lr']
            feedback = group['feedback']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if not stiefel:  # original procedure
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                            buf.mul_(momentum).add_(d_p)
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf
                    p.data.add_(-baselr, d_p)
                else:
                    # no weight decay
                    p_2d = p.data.view(p.shape[0], -1)
                    eye_p_2d = torch.eye(p_2d.shape[0], device=p.device)
                    inverse_approx_2d = eye_p_2d.mul(2).sub(p_2d.mm(p_2d.t()))
                    lr = baselr * stiefel
                    if momentum != 0:  # Riemannian momentum
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:  # v0
                            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                            buf.add_(d_p)
                            buf_2d = buf.view_as(p_2d)
                            q = buf_2d.mm(p_2d.t())
                            q = q.add(q.t()).mul(0.5)
                            buf_2d.sub_(q.mm(inverse_approx_2d).mm(p_2d))
                            buf.mul_(-lr)
                        else:
                            buf = param_state['momentum_buffer']
                            buf_2d = buf.view_as(p_2d)
                            
                            # C
                            con = torch.zeros_like(p.data)
                            con_2d = con.view_as(p_2d)
                            con_2d.sub_(buf_2d.mm(buf_2d.t()).mm(inverse_approx_2d).mm(p_2d))

                            # D
                            rd = torch.zeros_like(p.data)
                            rd_2d = rd.view_as(p_2d)
                            rd.add_(momentum - 1, buf).add_(-lr, d_p)
                            q = rd_2d.mm(p_2d.t())
                            rd_2d.sub_(q.add(q.t()).mul(0.5).mm(inverse_approx_2d).mm(p_2d))

                            # E
                            ext = torch.zeros_like(p.data)
                            ext_2d = ext.view_as(p_2d)
                            q = buf_2d.mm(p_2d.t())
                            ext_2d.add_(q.add(q.t()).mul(0.5).mm(q).mm(inverse_approx_2d).mm(inverse_approx_2d).mm(p_2d))

                            # F_phi
                            fb = torch.zeros_like(p.data)
                            fb_2d = fb.view_as(p_2d)
                            if feedback:
                                fb_2d.sub_(feedback, (inverse_approx_2d.mm(p_2d).mm(buf_2d.t()) + buf_2d.mm(p_2d.t())).mm(inverse_approx_2d).mm(p_2d))

                            buf.add_(con).add_(rd).add_(ext).add_(fb)

                        # no Nesterov
                        d_p = buf

                        # F_theta
                        fb = torch.zeros_like(p.data)
                        fb_2d = fb.view_as(p_2d)
                        if feedback:
                            fb_2d.sub_(feedback, p_2d.sub(inverse_approx_2d.mm(p_2d)))
                        p.data.add_(d_p).add_(fb)
                    else:  # no momentum
                        d_p.mul_(-lr)
                        d_p_2d = d_p.view_as(p_2d)
                        q = d_p_2d.mm(p_2d)
                        q = q.add(q.t()).mul(0.5)
                        d_p_2d.sub_(q.mm(inverse_approx_2d).mm(p_2d))
                        if feedback:
                            d_p_2d.sub_(feedback, p_2d.mm(p_2d.t()).mm(p_2d).sub(p_2d))
                        p.data.add_(d_p)
        return loss
