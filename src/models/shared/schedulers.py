import torch

class LinearBetaScheduler:
    def __init__(self, beta_start, beta_end, num_steps):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps
        self.beta = beta_start

    def update(self, step):
        step = min(step, self.num_steps)
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * (step / self.num_steps)
        return self.beta