import torch


a = torch.tensor([1, 2, 3, 4, 3.5])
f = 1.0 * a.sum() / 10.0
print("f = %f" % f)
