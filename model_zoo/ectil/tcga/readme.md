Download these models from [https://files.aiforoncology.nl/ectil](https://files.aiforoncology.nl/ectil)

These checkpoints are from the 5-fold training and evaluation runs on TCGA as presented in the related publication.

They are the `net` state_dicts saved from the `h5_module` and can be loaded with 

```py
weights = torch.load(
            "/path/to/checkpoint.ckpt"
            map_location=torch.device(device),
            weights_only=True,
        )
```

For the following `net` in the `h5_module`
```
MeanMIL(
    (post_encoder): Sequential(
      (0): Dropout(p=0, inplace=False)
      (1): Dropout1d(p=0, inplace=False)
      (2): Linear(in_features=2048, out_features=512, bias=True)
      (3): ReLU()
    )
    (classifier): Sequential(
      (0): Dropout(p=0.4, inplace=False)
      (1): Dropout1d(p=0.1, inplace=False)
      (2): Linear(in_features=512, out_features=1, bias=True)
      (3): Sigmoid()
    )
    (attention): GatedAttention(
      (attention): Linear(in_features=128, out_features=1, bias=True)
      (gate): Sequential(
        (0): Linear(in_features=512, out_features=128, bias=True)
        (1): Sigmoid()
      )
      (attention_hidden): Sequential(
        (0): Linear(in_features=512, out_features=128, bias=True)
        (1): Tanh()
      )
    )
)
```
With the following keys:
```
odict_keys(['net.post_encoder.2.weight', 'net.post_encoder.2.bias', 'net.classifier.2.weight', 'net.classifier.2.bias', 'net.attention.attention.weight', 'net.attention.attention.bias', 'net.attention.gate.0.weight', 'net.attention.gate.0.bias', 'net.attention.attention_hidden.0.weight', 'net.attention.attention_hidden.0.bias'])
```

To load the weights onto a plain torch model, remove the `net.` prefix from all the keys.

```py
weights_for_only_torch_model = {k.replace("net.", ""): v for k, v in weights.items()}
```

using the above `weights_for_only_torch_model`, the weights can be loaded onto a torch model and used outside of `lightning`, or passed to your own `lightning` modules. 
```py
from ectil.models.components import MeanMIL, GatedAttention
from torch.nn import Identity, Linear, Sequential, ReLU, Sigmoid

model = MeanMIL(
        post_encoder=Sequential(
            Identity(),
            Identity(),
            Linear(in_features=2048, out_features=512, bias=True),
            ReLU(),
        ),  # Replacing dropouts by Identities to match the index in sequential
        classifier=Sequential(
            Identity(),
            Identity(),
            Linear(in_features=512, out_features=1, bias=True),
            Sigmoid(),
        ),  # Replacing dropouts by Identities to match the index in sequential
        attention=GatedAttention(in_features=512, hidden_features=128),
    )
model.load_state_dict(weights_for_only_torch_model)
```