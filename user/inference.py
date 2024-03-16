import os
import sys
import torch
import numpy as np
import torch.nn.functional as F

from torch.nn.functional import softmax, interpolate

sys.path.append(os.getcwd())

from src.models import DeepLabV3plusModel
from src.utils.utils import crop_and_resize_tissue_patch, get_point_predictions

from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig,
    SegformerImageProcessor,
)


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
            num_channels=3,
            pretrained=pretrained_backbone,
            dropout_rate=dropout_rate,
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


class Deeplabv3TissueCellModel:
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """

    def __init__(self, metadata, cell_model_path: str, tissue_model_path: str):
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
        self.cell_branch = DeepLabV3plusModel(
            backbone_name=backbone_model,
            num_classes=3,
            num_channels=6,
            pretrained=pretrained_backbone,
            dropout_rate=dropout_rate,
        )
        self.cell_branch.load_state_dict(torch.load(cell_model_path))
        self.cell_branch.eval()
        self.cell_branch.to(self.device)

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


class SegFormerCellOnlyModel:
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """

    def __init__(self, metadata, model_path: str):
        self.metadata = metadata
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        configuration = SegformerConfig(
            num_labels=3,
            num_channels=3,
            depths=[3, 4, 18, 3],  # MiT-b3
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=768,
        )
        self.model = SegformerForSemanticSegmentation(configuration)
        self.model.load_state_dict(torch.load(model_path))
        self.image_processor = SegformerImageProcessor(
            do_resize=False, do_normalize=True
        )
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, cell_patch, tissue_patch, pair_id):
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
        # Getting the metadata corresponding to the patch pair ID
        meta_pair = self.metadata[pair_id]

        # Values are expected to be in the range [0, 255] for image_processor
        cell_patch = torch.tensor(cell_patch).permute(2, 0, 1).to(torch.uint8)
        preprocessed = self.image_processor.preprocess(cell_patch, return_tensors="pt")
        cell_patch = torch.tensor(preprocessed["pixel_values"])

        # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        # cell_patch = (cell_patch - mean) / std
        cell_patch = cell_patch.to(self.device)

        output = self.model(cell_patch).logits.squeeze(0).detach().cpu()
        output = interpolate(
            output.unsqueeze(0),
            size=cell_patch.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        output = output.squeeze(0)
        softmaxed = softmax(output, dim=0)
        result = get_point_predictions(softmaxed)
        return result
