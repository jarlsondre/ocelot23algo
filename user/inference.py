import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import softmax, interpolate
from typing import Dict, List, Union, Optional

sys.path.append(os.getcwd())

from src.models import DeepLabV3plusModel, CustomSegformerModel
from src.utils.utils import crop_and_resize_tissue_patch, get_point_predictions


def validate_inputs(cell_patch: np.ndarray, tissue_patch: np.ndarray) -> None:
    if not (cell_patch.shape == (1024, 1024, 3)):
        raise ValueError("Invalid shape for cell_patch")
    if not (tissue_patch.shape == (1024, 1024, 3)):
        raise ValueError("Invalid shape for tissue_patch")
    if not (cell_patch.dtype == np.uint8):
        raise ValueError("Invalid dtype for cell_patch")
    if not (tissue_patch.dtype == np.uint8):
        raise ValueError("Invalid dtype for tissue_patch")
    if not (cell_patch.min() >= 0 and cell_patch.max() <= 255):
        raise ValueError("Invalid value range for cell_patch")
    if not (tissue_patch.min() >= 0 and tissue_patch.max() <= 255):
        raise ValueError("Invalid value range for tissue_patch")


class Deeplabv3CellOnlyModel:
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """

    def __init__(
        self,
        metadata: Dict,
        cell_model: Union[str, nn.Module],
        tissue_model_path: Optional[str] = None,
    ):

        assert tissue_model_path is None

        self.metadata = metadata
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone_model = "resnet50"
        dropout_rate = 0.3
        pretrained_backbone = True

        self.model: nn.Module
        if isinstance(cell_model, str):
            self.model = DeepLabV3plusModel(
                backbone_name=backbone_model,
                num_classes=3,
                num_channels=3,
                pretrained=pretrained_backbone,
                dropout_rate=dropout_rate,
            )
            self.model.load_state_dict(torch.load(cell_model))
        elif isinstance(cell_model, torch.nn.Module):
            self.model = cell_model
        else:
            raise ValueError("Invalid cell model type")

        self.model.eval()
        self.model.to(self.device)

    def __call__(self, cell_patch, tissue_patch, pair_id, transform=None):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8]
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        validate_inputs(cell_patch, tissue_patch)
        if transform:
            transformed = transform(image=cell_patch)
            cell_patch = transformed["image"]

        # Scaling to [0, 1] if needed
        if cell_patch.dtype == np.uint8:
            cell_patch = cell_patch.astype(np.float32) / 255.0
        elif cell_patch.dtype != np.float32:
            cell_patch = cell_patch.astype(np.float32)

        # Preparing shape and device for model input
        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0).to(self.device)

        # Getting model output and processing
        output = self.model(cell_patch).squeeze(0).detach().cpu()
        softmaxed = softmax(output, dim=0)
        result = get_point_predictions(softmaxed)
        return result


class Deeplabv3TissueCellModel:
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """

    def __init__(
        self, metadata: List, cell_model: Union[str, nn.Module], tissue_model_path: str
    ):
        self.metadata = metadata
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone_model = "resnet50"
        dropout_rate = 0.3
        pretrained_backbone = True

        # Create tissue branch
        self.tissue_branch = DeepLabV3plusModel(
            backbone_name=backbone_model,
            num_classes=3,
            num_channels=3,
            pretrained=pretrained_backbone,
            dropout_rate=dropout_rate,
        )
        self.tissue_branch.load_state_dict(torch.load(tissue_model_path))
        self.tissue_branch.eval()
        self.tissue_branch.to(self.device)

        # Create cell branch
        self.cell_branch: nn.Module
        if isinstance(cell_model, str):
            self.cell_branch = DeepLabV3plusModel(
                backbone_name=backbone_model,
                num_classes=3,
                num_channels=6,
                pretrained=pretrained_backbone,
                dropout_rate=dropout_rate,
            )
            self.cell_branch.load_state_dict(torch.load(cell_model))
        elif isinstance(cell_model, torch.nn.Module):
            self.cell_branch = cell_model
        else:
            raise ValueError("Invalid cell model type")
        self.cell_branch.eval()
        self.cell_branch.to(self.device)

    def __call__(
        self,
        cell_patch: np.ndarray,
        tissue_patch: np.ndarray,
        pair_id: int,
        transform=None,
    ):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided.

        Parameters
        ----------
        cell_patch:
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch:
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id:
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        # Getting the metadata corresponding to the patch pair ID
        meta_pair = self.metadata[pair_id]
        tissue_mpp = meta_pair["tissue"]["resized_mpp_x"]
        cell_mpp = meta_pair["cell"]["resized_mpp_x"]
        x_offset = meta_pair["patch_x_offset"]
        y_offset = meta_pair["patch_y_offset"]

        validate_inputs(cell_patch, tissue_patch)

        if transform:
            transformed = transform(image=cell_patch, tissue=tissue_patch)
            cell_patch = transformed["image"]
            tissue_patch = transformed["tissue"]

        # Scaling to [0, 1] if needed
        if cell_patch.dtype == np.uint8:
            cell_patch = cell_patch.astype(np.float32) / 255.0
        elif cell_patch.dtype != np.float32:
            cell_patch = cell_patch.astype(np.float32)

        if tissue_patch.dtype == np.uint8:
            tissue_patch = tissue_patch.astype(np.float32) / 255.0
        elif tissue_patch.dtype != np.float32:
            tissue_patch = tissue_patch.astype(np.float32)

        # Preparing shape and device for model input
        tissue_patch: torch.Tensor
        tissue_patch = torch.from_numpy(tissue_patch).permute(2, 0, 1)
        tissue_patch = tissue_patch.unsqueeze(0).to(self.device)

        # Predicting tissue
        tissue_prediction = self.tissue_branch(tissue_patch).squeeze(0)
        argmaxed = tissue_prediction.argmax(dim=0)

        cropped_tissue: torch.Tensor = crop_and_resize_tissue_patch(
            image=argmaxed,
            tissue_mpp=tissue_mpp,
            cell_mpp=cell_mpp,
            x_offset=x_offset,
            y_offset=y_offset,
        )

        one_hot_cropped_tissue = F.one_hot(cropped_tissue, num_classes=3).permute(
            2, 0, 1
        )
        one_hot_cropped_tissue = one_hot_cropped_tissue.unsqueeze(0).to(self.device)

        # Cell Patch
        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0).to(self.device)

        # Concatenating to create final input
        model_input = torch.cat([cell_patch, one_hot_cropped_tissue], dim=1)

        # Getting prediction
        cell_prediction = self.cell_branch(model_input).squeeze(0).detach().cpu()
        softmaxed = softmax(cell_prediction, dim=0)
        result = get_point_predictions(softmaxed)
        return result


class Deeplabv3TissueFromFile:
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """

    def __init__(self, metadata, cell_model_path: str, tissue_model_path=None):
        # Just to make it easier to swap models in the other file
        assert tissue_model_path is None

        self.metadata = metadata
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone_model = "resnet50"
        dropout_rate = 0.3
        pretrained_backbone = True

        self.model = DeepLabV3plusModel(
            backbone_name=backbone_model,
            num_classes=3,
            num_channels=6,
            pretrained=pretrained_backbone,
            dropout_rate=dropout_rate,
        )
        self.model.load_state_dict(torch.load(cell_model_path))
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, cell_patch, tissue_patch, pair_id, transform=None) -> list:
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8]
            Tissue patch with shape [1024, 1024, 3] with values in {0, 1}
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        # Getting the metadata corresponding to the patch pair ID
        validate_inputs(cell_patch, tissue_patch)
        if transform:
            transformed = transform(image=cell_patch, tissue=tissue_patch)
            cell_patch = transformed["image"]
            tissue_patch = transformed["tissue"]

        # Making sure the range is [0, 1] if no
        if cell_patch.dtype == np.uint8:
            cell_patch = cell_patch.astype(np.float32) / 255.0
        elif cell_patch.dtype != np.float32:
            cell_patch = cell_patch.astype(np.float32)

        if tissue_patch.dtype != np.float32:
            tissue_patch = tissue_patch.astype(np.float32)

        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0).to(self.device)

        tissue_patch = torch.from_numpy(tissue_patch).permute(2, 0, 1)
        tissue_patch = tissue_patch.unsqueeze(0).to(self.device)

        model_input = torch.cat([cell_patch, tissue_patch], dim=1)

        cell_prediction = self.model(model_input).squeeze(0).detach().cpu()
        softmaxed = softmax(cell_prediction, dim=0)
        result = get_point_predictions(softmaxed)
        return result


class SegformerCellOnlyModel:
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """

    def __init__(self, metadata, cell_model_path: str, tissue_model_path=None):
        # Just to make it easier to swap models in the other file
        assert tissue_model_path is None

        self.metadata = metadata
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone_model = "b3"

        self.model = CustomSegformerModel(
            backbone_name=backbone_model,
            num_classes=3,
            num_channels=3,
        )
        self.model.load_state_dict(torch.load(cell_model_path))
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, cell_patch, tissue_patch, pair_id, transform=None):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8]
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        validate_inputs(cell_patch, tissue_patch)
        if transform:
            transformed = transform(image=cell_patch)
            cell_patch = transformed["image"]

        # Scaling to [0, 1] if needed
        if cell_patch.dtype == np.uint8:
            cell_patch = cell_patch.astype(np.float32) / 255.0
        elif cell_patch.dtype != np.float32:
            cell_patch = cell_patch.astype(np.float32)

        # Preparing shape and device for model input
        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0).to(self.device)

        # Getting model output and processing
        output = self.model(cell_patch).squeeze(0).detach().cpu()
        softmaxed = softmax(output, dim=0)
        result = get_point_predictions(softmaxed)
        return result


class SegformerTissueFromFile:
    """
    A Segformer evaluate model that uses tissue images from file and cell images
    from the input.

    Args:
        metadata: A dictionary containing metadata about the dataset.
        cell_model: A path to the pre-trained model (str) or a pre-loaded PyTorch model (Module).
                    The model is used for cell segmentation within tissue images.
        tissue_model_path: Not used for this model, must be None
    """

    def __init__(
        self,
        metadata: Dict,
        cell_model: Union[str, nn.Module],
        tissue_model_path: Optional[str] = None,
    ):
        assert tissue_model_path is None

        self.metadata = metadata
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone_model = "b3"

        self.model: nn.Module
        if isinstance(cell_model, str):
            self.model = CustomSegformerModel(
                backbone_name=backbone_model,
                num_classes=3,
                num_channels=6,
            )
            self.model.load_state_dict(torch.load(cell_model))
        elif isinstance(cell_model, torch.nn.Module):
            self.model = cell_model
        else:
            raise ValueError("Invalid cell model type")

        self.model.eval()
        self.model.to(self.device)

    def __call__(self, cell_patch, tissue_patch, pair_id, transform=None) -> list:
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8]
            Tissue patch with shape [1024, 1024, 3] with values in {0, 1}
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        # Getting the metadata corresponding to the patch pair ID
        validate_inputs(cell_patch, tissue_patch)
        if transform:
            transformed = transform(image=cell_patch, tissue=tissue_patch)
            cell_patch = transformed["image"]
            tissue_patch = transformed["tissue"]

        # Making sure the range is [0, 1] if no
        if cell_patch.dtype == np.uint8:
            cell_patch = cell_patch.astype(np.float32) / 255.0
        elif cell_patch.dtype != np.float32:
            cell_patch = cell_patch.astype(np.float32)

        if tissue_patch.dtype != np.float32:
            tissue_patch = tissue_patch.astype(np.float32)

        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0).to(self.device)

        tissue_patch = torch.from_numpy(tissue_patch).permute(2, 0, 1)
        tissue_patch = tissue_patch.unsqueeze(0).to(self.device)

        model_input = torch.cat([cell_patch, tissue_patch], dim=1)

        cell_prediction = self.model(model_input).squeeze(0).detach().cpu()
        softmaxed = softmax(cell_prediction, dim=0)
        result = get_point_predictions(softmaxed)
        return result
