import torch


class BaseSampler():
    def sample(self, n_samples=1):
        raise NotImplementedError

    def sample_batch(self, n_samples=1):
        raise NotImplementedError


class RandomSampler():
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def sample(self, n_samples=1):
        return torch.randint(self.num_samples, size=(n_samples,)).long()

    def sample_batch(self):
        return torch.arange(self.num_samples).long()


class ShuffleSampler():
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.permutation = torch.randperm(self.num_samples)
        self.pointer = 0

    def sample(self, n_samples=1):
        if self.pointer >= self.num_samples:
            self.pointer = 0
            self.permutation = torch.randperm(self.num_samples)

        batch = self.permutation[self.pointer:self.pointer+n_samples].long()
        self.pointer += n_samples
        return batch

    def sample_batch(self):
        return torch.arange(self.num_samples).long()