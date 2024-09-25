from pathlib import Path

import torch
from torch.nn import Identity, Linear, ReLU, Sequential, Sigmoid
from torchvision import transforms

import ectil
from ectil.models.components import GatedAttention, MeanMIL, RetCCL

# Minimal example to run inference on a set of images. This assumes that
# the user has extracted images of interest from a WSI.
# Run as module from project root dir, i.e.
# ~/ectil$ python -m tools.infer.minimal_example


def instantiate_ectil():
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
    ).eval()
    return model


def load_weights_from_state_dict():
    path_to_state_dict = (
        Path(ectil.__path__[0]).parent
        / "model_zoo/ectil/tcga/fold_0/epoch_065_step_858_weights_only.ckpt"
    )

    use_gpu = torch.cuda.is_available()

    weights_for_only_torch_model = {
        k.replace("net.", ""): v
        for k, v in torch.load(
            path_to_state_dict,
            weights_only=True,
            map_location=torch.device("cuda") if use_gpu else torch.device("cpu"),
        ).items()
    }

    return weights_for_only_torch_model


def get_transform():
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return transform


def instantiate_feature_encoder():
    return RetCCL(
        project_root_dir=""
    ).eval()  # Makes the path relative to the project root


if __name__ == "__main__":

    # Get the ECTIL weights from model_zoo/ectil/tcga/*/*.ckpt
    ectil_weights = load_weights_from_state_dict()
    print(f"Loaded ECTIL weights: \n{ectil_weights.keys()}\n")

    # Define RetCCL feature encoder; a ResNet50 w/ a torch.nn.Identity() instead of the original classifier
    encoder = instantiate_feature_encoder()
    print(
        f"Loaded encoder, a ResNet50 with only a removed classification layer which has an identity instead: encoder.net.fc={encoder.net.fc}\n"
    )

    # Define TILs regression model
    ectil = instantiate_ectil()
    print(f"Loaded ECTIL: \n{ectil}\n")

    # Apply the weights
    ectil.load_state_dict(ectil_weights)

    # Define dummy input; a batch of WSIs with a collection of a varying number of
    # 3x512x512 RGB images with c*h*w rgb values in [0.,1.]
    # This collection are the patches of interest extracted from a single WSI and
    # may essentially be any number. Here is an example with, e.g., 1 WSI with 9 patches,
    # In practice, however, a WSI may have hundreds or thousands patches of interest
    # RetCCL expects a float32 torch.tensor of batch_size * channels * height * width
    dummy_images = torch.rand(
        size=(9, 3, 512, 512)
    )  # 1 WSI with 9 patches of 3*512*512
    print(f"Generated dummy images of the following shape: {dummy_images.shape}\n")

    # Get a transform for the images that RetCCL expects (imagenet normalization and [0,1] input)
    transform = get_transform()

    with torch.no_grad():
        # Compute features from image input
        retccl_features = encoder(transform(dummy_images))
        print(
            f"Extracted features from dummy images with shape {retccl_features.shape}\n"
        )

        # Compute TILs score from RetCCL features. .unsqueeze(0) adds the batch (# WSIs) dimension on position 0,
        # Which ECTIL expects
        # "out" contains the final TILs score ([0.,1.]), "out_per_instance" provides the TILs score per patch,
        # "attention_weights" contains the attention weight per patchs (sum up to 1)
        batched_retccl_features = retccl_features.unsqueeze(0)
        tils_score = ectil(batched_retccl_features)
        print(
            f"Computed TILs score for dummy input of shape {batched_retccl_features.shape}:\ntils={tils_score}\n"
        )
