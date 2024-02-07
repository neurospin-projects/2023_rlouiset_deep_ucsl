import torch

@torch.no_grad()
class PytorchStandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        self.mean = torch.mean(values, dim=0, keepdim=True)
        self.std = torch.std(values, dim=0, keepdim=True)
        return self

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

@torch.no_grad()
class PytorchRobustScaler:
    def __init__(self, median=None, iqr=None, epsilon=1e-7, quantiles=None):
        """Robust Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param median: The median of the features. The property will be set after a call to fit.
        :param inter_quantile_range: The iqr of the features. The property will be set after a call to fit.
        :param quantiles: quantiles of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        if quantiles is None:
            quantiles = [0.25, 0.75]
        self.inter_quantile_range = iqr
        self.reference_quantiles = torch.tensor(quantiles)
        self.median = median
        self.quantiles = None
        self.epsilon = epsilon

    def fit(self, values):
        self.quantiles = torch.quantile(values, self.reference_quantiles.to(values.get_device()), dim=0, keepdim=True)
        self.inter_quantile_range = (self.quantiles[1] - self.quantiles[0]).abs()
        self.median = torch.median(values, dim=0, keepdim=True).values
        return self

    def transform(self, values):
        return (values - self.median) / (self.inter_quantile_range + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)