import numpy as np
import torch
from torch import nn
from utils.nn.mlp import MLP
from utils.nn.rff_mlp import RFFMLP
from utils.nn.siren import SirenNet
from utils.pe.projection import Projection
from utils.pe.projection_rff import ProjectionRFF
from utils.pe.spherical_harmonics import SphericalHarmonics


def get_positional_encoding(positional_encoding_type, hparams, device="cuda"):
    """
    Returns a positional encoding module based on the specified encoding type.

    Args:
        encoding_type (str): The type of positional encoding to use. Options are 'rff', 'siren', 'sh', 'capsule'.
        input_dim (int): The input dimension for the positional encoding.
        output_dim (int): The output dimension for the positional encoding.
        hparams: Additional arguments for specific encoding types.

    Returns:
        nn.Module: The positional encoding module.
    """
    if positional_encoding_type == "projectionrff":
        return ProjectionRFF(
            projection=hparams["projection"],
            sigma=hparams["sigma"],
            hparams=hparams,
            device=device,
        )
    elif positional_encoding_type == "projection":
        return Projection(
            projection=hparams["projection"], hparams=hparams, device=device
        )
    elif positional_encoding_type == "sh":
        return SphericalHarmonics(
            legendre_polys=hparams["legendre_polys"],
            harmonics_calculation=hparams["harmonics_calculation"],
            hparams=hparams,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported encoding type: {positional_encoding_type}")


def get_neural_network(neural_network_type, input_dim, hparams=None, device="cuda"):
    """
    Returns a neural network module based on the specified network type.

    Args:
        neural_network_type (str): The type of neural network to use. Options are 'siren'.
        input_dim (int): The input dimension for the neural network.
        output_dim (int): The output dimension for the neural network.
        hparams: Additional arguments for specific network types.

    Returns:
        nn.Module: The neural network module.
    """
    if neural_network_type == "siren":
        return SirenNet(
            input_dim=input_dim,
            output_dim=hparams["output_dim"],
            hidden_dim=hparams["hidden_dim"],
            num_layers=hparams["num_layers"],
            hparams=hparams,
            device=device,
        )
    elif neural_network_type == "mlp":
        return MLP(
            input_dim=input_dim,
            hidden_dim=hparams["hidden_dim"],
            hparams=hparams,
            device=device,
        )
    elif neural_network_type == "rffmlp":
        return RFFMLP(
            input_dim=input_dim,
            hidden_dim=hparams["hidden_dim"],
            sigma=hparams["sigma"],
            hparams=hparams,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported network type: {neural_network_type}")


class LocationEncoder(nn.Module):
    def __init__(
        self,
        positional_encoding_type="sh",
        neural_network_type="siren",
        hparams=None,
        device="cuda",
    ):
        super().__init__()
        self.device = device

        self.position_encoder = get_positional_encoding(
            positional_encoding_type=positional_encoding_type,
            hparams=hparams,
            device=device,
        )

        self.neural_network = nn.ModuleList(
            [
                get_neural_network(
                    neural_network_type, input_dim=dim, hparams=hparams, device=device
                )
                for dim in self.position_encoder.embedding_dim
            ]
        )

    def forward(self, x):
        embedding = self.position_encoder(x)

        if embedding.ndim == 2:
            # If the embedding is (batch, n), we need to add a dimension
            embedding = embedding.unsqueeze(0)

        location_features = torch.zeros(embedding.shape[1], 512).to(self.device)

        for nn, e in zip(self.neural_network, embedding):
            location_features += nn(e)

        return location_features
