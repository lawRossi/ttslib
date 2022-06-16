import torch
import numpy as np


class ScheduledOptimizer:

    def __init__(self, model, train_config):
        self.model = model
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
        self.current_step = 0
        self.init_lr = train_config["optimizer"]["init_lr"]
        self.train_config = train_config

    def step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_thresh)
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()
    
    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)
    
    def get_lr(self):
        return self._optimizer.param_groups[0]["lr"]

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


class FineTuningOptimizer:
    def __init__(self, model, train_config):
        self.model = model
        keys = ["embedding.weight"]
        parameters = [params for name, params in self.model.named_parameters() 
                      if not any(key in name for key in keys)]
        self._optimizer = torch.optim.Adam(
            parameters,
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"]
        )
        self.init_lr = train_config["optimizer"]["fine_tuning_lr"]
        self.grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
        self.current_step = 0

    def step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_thresh)
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)
    
    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr_scale = np.power(self.current_step, -0.5)
        lr = self.init_lr * lr_scale

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self._optimizer.param_groups[0]["lr"]
