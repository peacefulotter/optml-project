

import torch

a = torch.empty((2, 2))
a[0, 0] = 1
a[0, 1] = 2
a[1, 0] = 3
a[1, 1] = 4
print(a, torch.flatten(a))