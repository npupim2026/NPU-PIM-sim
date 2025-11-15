# custom_torch

Makes simple to use torch-implemented model.

Just replace the imports as below and do slight modification, to run on MIDAP
```python
import models.custom_torch as torch
import models.custom_torch.newnetwork as nn
import models.custom_torch.newnetwork as F
```

## things to modify
* Make sure the function to obtain model returns nn.mb. refer to get_densenet ftn of examples/densenet.py.
```python
input_size = 224
input_shape = (1, 3, input_size, input_size)

nn.reset_mb()
x = nn.set_input_tensor(tensor_shape=input_shape)
x = net(x)
return nn.mb
```
* modify + to torch.add
* remove x.view or torch.flatten, torch.init, etc...

## Note
I intended to build only one network using the module.
