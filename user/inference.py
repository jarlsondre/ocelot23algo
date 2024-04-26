import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from torch.nn.functional import softmax
from typing import Dict, List, Union, Optional

sys.path.append(os.getcwd())

from src.models import DeepLabV3plusModel, CustomSegformerModel
from src.models import (
    SegformerSharingModel as SegformerSharingModule,
    SegformerTissueToCellDecoderModel as SegformerTissueToCellDecoderModule,
)
from src.utils.utils import crop_and_resize_tissue_patch, get_point_predictions


class EvaluationModel(ABC):

    metadata: Dict
    cell_model: nn.Module
    device: torch.device
    backbone_model: Optional[str]
    tissue_from_file: bool

    @abstractmethod
    def __init__(
        self,
        metadata: Dict,
        cell_model: Union[str, nn.Module],
        tissue_model_path: Optional[str] = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.metadata = metadata
        self.device = device
        self.tissue_from_file = False

    @abstractmethod
    def __call__(self, cell_patch, tissue_patch, pair_id, transform=None) -> List:
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8]
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255 or
            {0, 1}, depending on the model
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        pass

    # TODO: This does exactly the same as the function below...
    def _scale_cell_patch(self, cell_patch: np.ndarray) -> np.ndarray:
        """
        Scales the cell_patch to [0, 1] if it is of type uint8, and converts
        it to float32 if it is not already.
        """
        if cell_patch.dtype == np.uint8:
            cell_patch = cell_patch.astype(np.float32) / 255.0
        elif cell_patch.dtype != np.float32:
            cell_patch = cell_patch.astype(np.float32)

        return cell_patch

    def _scale_tissue_patch(
        self, tissue_patch: np.ndarray, do_scale: bool = True
    ) -> np.ndarray:
        """
        Scales the tissue_patch to [0, 1] if do_scale is true and if dtype is
        uint8, and converts it to float32 if it is not already.
        """
        if do_scale and tissue_patch.dtype == np.uint8:
            tissue_patch = tissue_patch.astype(np.float32) / 255.0
        elif tissue_patch.dtype != np.float32:
            tissue_patch = tissue_patch.astype(np.float32)

        return tissue_patch

    def validate_inputs(self, cell_patch: np.ndarray, tissue_patch: np.ndarray) -> None:
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

        if self.tissue_from_file:
            # Must be in {0, 1}
            if not (tissue_patch.min() in [0, 1] and tissue_patch.max() in [0, 1]):
                raise ValueError("Invalid value range for tissue_patch, must be {0, 1}")
        else:
            # Must be [0, 255]
            if not (tissue_patch.min() >= 0 and tissue_patch.max() <= 255):
                raise ValueError(
                    "Invalid value range for tissue_patch, must be [0, 255]"
                )


class Deeplabv3CellOnlyModel(EvaluationModel):

    def __init__(self, metadata, cell_model, tissue_model_path=None):
        super().__init__(metadata, cell_model, tissue_model_path)
        backbone_model = "resnet50"
        dropout_rate = 0.3
        pretrained_backbone = True
        self.tissue_from_file = False

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
        self.validate_inputs(cell_patch, tissue_patch)
        self.model.eval()

        if transform is not None:
            transformed = transform(image=cell_patch)
            cell_patch = transformed["image"]

        cell_patch = self._scale_cell_patch(cell_patch)

        # Preparing shape and device for model input
        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0).to(self.device)

        # Getting model output and processing
        output = self.model(cell_patch).squeeze(0).detach().cpu()
        softmaxed = softmax(output, dim=0)
        result = get_point_predictions(softmaxed)
        return result


class Deeplabv3TissueCellModel(EvaluationModel):

    def __init__(self, metadata, cell_model, tissue_model_path):
        super().__init__(metadata, cell_model, tissue_model_path)
        backbone_model = "resnet50"
        dropout_rate = 0.3
        pretrained_backbone = True
        self.tissue_from_file = False

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

    def __call__(self, cell_patch, tissue_patch, pair_id, transform=None):
        self.tissue_branch.eval()
        self.cell_branch.eval()

        # Getting the metadata corresponding to the patch pair ID
        meta_pair = self.metadata[pair_id]
        tissue_mpp = meta_pair["tissue"]["resized_mpp_x"]
        cell_mpp = meta_pair["cell"]["resized_mpp_x"]
        x_offset = meta_pair["patch_x_offset"]
        y_offset = meta_pair["patch_y_offset"]

        self.validate_inputs(cell_patch, tissue_patch)

        if transform:
            transformed = transform(image=cell_patch, tissue=tissue_patch)
            cell_patch = transformed["image"]
            tissue_patch = transformed["tissue"]

        # Scaling to [0, 1] if needed
        cell_patch = self._scale_cell_patch(cell_patch)
        tissue_patch = self._scale_tissue_patch(tissue_patch, do_scale=True)

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


class Deeplabv3TissueFromFile(EvaluationModel):

    def __init__(self, metadata, cell_model, tissue_model_path=None):
        assert tissue_model_path is None
        super().__init__(metadata, cell_model, tissue_model_path)
        backbone_model = "resnet50"
        dropout_rate = 0.3
        pretrained_backbone = True
        self.tissue_from_file = True

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
        """
        Note:
        - Expects tissue_patch to have values in {0, 1}

        """
        self.validate_inputs(cell_patch, tissue_patch)
        self.model.eval()

        if transform is not None:
            transformed = transform(image=cell_patch, tissue=tissue_patch)
            cell_patch = transformed["image"]
            tissue_patch = transformed["tissue"]

        cell_patch = self._scale_cell_patch(cell_patch)
        tissue_patch = self._scale_tissue_patch(tissue_patch, do_scale=False)

        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0).to(self.device)

        tissue_patch = torch.from_numpy(tissue_patch).permute(2, 0, 1)
        tissue_patch = tissue_patch.unsqueeze(0).to(self.device)

        model_input = torch.cat([cell_patch, tissue_patch], dim=1)

        cell_prediction = self.model(model_input).squeeze(0).detach().cpu()
        softmaxed = softmax(cell_prediction, dim=0)
        result = get_point_predictions(softmaxed)
        return result


class SegformerCellOnlyModel(EvaluationModel):
    def __init__(self, metadata, cell_model, tissue_model_path=None):
        # Just to make it easier to swap models in the other file
        assert tissue_model_path is None
        super().__init__(metadata, cell_model, tissue_model_path)
        backbone_model = "b3"
        self.tissue_from_file = False

        if isinstance(cell_model, str):
            self.model = CustomSegformerModel(
                backbone_name=backbone_model,
                num_classes=3,
                num_channels=3,
            )
            self.model.load_state_dict(torch.load(cell_model))
        elif isinstance(cell_model, torch.nn.Module):
            self.model = cell_model
        else:
            raise ValueError("Invalid cell model type")

        self.model.eval()
        self.model.to(self.device)

    def __call__(self, cell_patch, tissue_patch, pair_id, transform=None):
        self.validate_inputs(cell_patch, tissue_patch)
        if transform is not None:
            transformed = transform(image=cell_patch, mask=cell_patch)
            cell_patch = transformed["image"]

        cell_patch = self._scale_cell_patch(cell_patch)

        # Preparing shape and device for model input
        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0).to(self.device)

        # Getting model output and processing
        output = self.model(cell_patch).squeeze(0).detach().cpu()
        softmaxed = softmax(output, dim=0)
        result = get_point_predictions(softmaxed)
        return result


class SegformerTissueFromFile(EvaluationModel):

    def __init__(
        self, metadata, cell_model, tissue_model_path=None, device=torch.device("cuda")
    ):
        assert tissue_model_path is None
        super().__init__(metadata, cell_model, tissue_model_path, device=device)
        backbone_model = "b3"
        self.tissue_from_file = True

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
        """
        Note:
        - Expects tissue_patch to have values in {0, 1}

        """
        self.validate_inputs(cell_patch, tissue_patch)
        if transform is not None:
            transformed = transform(
                image=cell_patch, tissue_prediction=tissue_patch, cell_label=None
            )
            cell_patch = transformed["image"]
            tissue_patch = transformed["tissue_prediction"]

        cell_patch = self._scale_cell_patch(cell_patch)
        tissue_patch = self._scale_tissue_patch(tissue_patch, do_scale=False)

        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0).to(self.device)

        tissue_patch = torch.from_numpy(tissue_patch).permute(2, 0, 1)
        tissue_patch = tissue_patch.unsqueeze(0).to(self.device)

        model_input = torch.cat([cell_patch, tissue_patch], dim=1)

        cell_prediction = self.model(model_input).squeeze(0).detach().cpu()
        softmaxed = softmax(cell_prediction, dim=0)
        result = get_point_predictions(softmaxed)
        return result


class SegformerSharingTissueFromFile(EvaluationModel):

    def __init__(
        self, metadata, cell_model, tissue_model_path=None, device=torch.device("cuda")
    ):
        assert tissue_model_path is None
        super().__init__(metadata, cell_model, tissue_model_path, device=device)
        backbone_model = "b3"
        self.tissue_from_file = True

        if isinstance(cell_model, str):
            self.model = CustomSegformerModel(  # TODO: Is this the correct model?
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
        """
        Note:
        - Expects tissue_patch to have values in {0, 1}

        """
        self.validate_inputs(cell_patch, tissue_patch)
        if transform is not None:
            transformed = transform(image=cell_patch, tissue=tissue_patch)
            cell_patch = transformed["image"]
            tissue_patch = transformed["tissue"]

        cell_patch = self._scale_cell_patch(cell_patch)
        tissue_patch = self._scale_tissue_patch(tissue_patch, do_scale=False)

        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0).to(self.device)

        tissue_patch = torch.from_numpy(tissue_patch).permute(2, 0, 1)
        tissue_patch = tissue_patch.unsqueeze(0).to(self.device)

        model_input = torch.cat([cell_patch, tissue_patch], dim=1)

        cell_prediction = self.model(model_input, (pair_id,)).squeeze(0).detach().cpu()
        softmaxed = softmax(cell_prediction, dim=0)[:3]
        result = get_point_predictions(softmaxed)
        return result


class SegformerSharingModel(EvaluationModel):

    def __init__(self, metadata, cell_model, tissue_model_path=None):
        assert tissue_model_path is None
        super().__init__(metadata, cell_model, tissue_model_path)
        backbone_model = "b3"

        if isinstance(cell_model, str):
            self.model = SegformerSharingModule(
                backbone_model=backbone_model,
                pretrained_dataset="ade",
                input_image_size=1024,
                output_image_size=1024,
            )
            self.model.load_state_dict(torch.load(cell_model))
        elif isinstance(cell_model, torch.nn.Module):
            self.model = cell_model
        else:
            raise ValueError("Invalid model type ")

    def __call__(self, cell_patch, tissue_patch, pair_id, transform=None) -> List:
        self.validate_inputs(cell_patch, tissue_patch)
        self.model.eval()

        # Getting the correct metadata
        meta_pair = self.metadata[pair_id]
        x_offset = meta_pair["patch_x_offset"]
        y_offset = meta_pair["patch_y_offset"]

        if transform is not None:
            transformed = transform(
                cell_image=cell_patch,
                tissue_image=tissue_patch,
                cell_label=cell_patch,  # placeholder
                tissue_label=cell_patch,  # placeholder
            )
            cell_patch = transformed["cell_image"]
            tissue_patch = transformed["tissue_image"]

        cell_patch = self._scale_cell_patch(cell_patch)
        tissue_patch = self._scale_tissue_patch(tissue_patch, do_scale=True)

        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0).to(self.device)

        tissue_patch = torch.from_numpy(tissue_patch).permute(2, 0, 1)
        tissue_patch = tissue_patch.unsqueeze(0).to(self.device)

        model_input = torch.cat([cell_patch, tissue_patch], dim=1)
        offsets = torch.tensor([[x_offset, y_offset]])

        cell_prediction, _ = self.model(model_input, offsets)
        cell_prediction = cell_prediction.squeeze(0).detach().cpu()
        softmaxed = softmax(cell_prediction, dim=0)

        result = get_point_predictions(softmaxed)
        return result


class SegformerTissueToCellDecoderModel(SegformerSharingModel):

    def __init__(self, metadata, cell_model, tissue_model_path=None):
        assert tissue_model_path is None
        super().__init__(metadata, cell_model, tissue_model_path)
        backbone_model = "b3"

        if isinstance(cell_model, str):
            self.model = SegformerTissueToCellDecoderModule(
                backbone_model=backbone_model,
                pretrained_dataset="ade",
                input_image_size=1024,
                output_image_size=1024,
            )
            self.model.load_state_dict(torch.load(cell_model))
        elif isinstance(cell_model, torch.nn.Module):
            self.model = cell_model
        else:
            raise ValueError("Invalid model type ")
