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
    SegformerJointPred2InputModel as SegformerSharingModule,
    SegformerAdditiveJointPred2DecoderModel as SegformerTissueToCellDecoderModule,
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

    def _scale_patch(self, patch: np.ndarray, do_scale: bool = True) -> np.ndarray:
        """
        Scales the to [0, 1] if do_scale is true and if dtype is
        uint8, and converts it to float32 if it is not already.
        """
        if do_scale and patch.dtype == np.uint8:
            patch = patch.astype(np.float32) / 255.0
        elif patch.dtype != np.float32:
            patch = patch.astype(np.float32)

        return patch

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

        cell_patch = self._scale_patch(cell_patch)

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
        cell_patch = self._scale_patch(cell_patch)
        tissue_patch = self._scale_patch(tissue_patch, do_scale=True)

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

        cell_patch = self._scale_patch(cell_patch)
        tissue_patch = self._scale_patch(tissue_patch, do_scale=False)

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

        cell_patch = self._scale_patch(cell_patch)

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

        cell_patch = self._scale_patch(cell_patch)
        tissue_patch = self._scale_patch(tissue_patch, do_scale=False)

        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0).to(self.device)

        tissue_patch = torch.from_numpy(tissue_patch).permute(2, 0, 1)
        tissue_patch = tissue_patch.unsqueeze(0).to(self.device)

        model_input = torch.cat([cell_patch, tissue_patch], dim=1)

        cell_prediction = self.model(model_input).squeeze(0).detach().cpu()
        softmaxed = softmax(cell_prediction, dim=0)
        result = get_point_predictions(softmaxed)
        return result


# class SegformerSharingTissueFromFile(EvaluationModel):

#     def __init__(
#         self, metadata, cell_model, tissue_model_path=None, device=torch.device("cuda")
#     ):
#         assert tissue_model_path is None
#         super().__init__(metadata, cell_model, tissue_model_path, device=device)
#         backbone_model = "b3"
#         self.tissue_from_file = True

#         if isinstance(cell_model, str):
#             self.model = CustomSegformerModel(  # TODO: Is this the correct model?
#                 backbone_name=backbone_model,
#                 num_classes=3,
#                 num_channels=6,
#             )
#             self.model.load_state_dict(torch.load(cell_model))
#         elif isinstance(cell_model, torch.nn.Module):
#             self.model = cell_model
#         else:
#             raise ValueError("Invalid cell model type")

#         self.model.eval()
#         self.model.to(self.device)

#     def __call__(self, cell_patch, tissue_patch, pair_id, transform=None) -> list:
#         """
#         Note:
#         - Expects tissue_patch to have values in {0, 1}

#         """
#         self.validate_inputs(cell_patch, tissue_patch)
#         if transform is not None:
#             transformed = transform(image=cell_patch, tissue=tissue_patch)
#             cell_patch = transformed["image"]
#             tissue_patch = transformed["tissue"]

#         cell_patch = self._scale_patch(cell_patch)
#         tissue_patch = self._scale_patch(tissue_patch, do_scale=False)

#         cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
#         cell_patch = cell_patch.unsqueeze(0).to(self.device)

#         tissue_patch = torch.from_numpy(tissue_patch).permute(2, 0, 1)
#         tissue_patch = tissue_patch.unsqueeze(0).to(self.device)

#         model_input = torch.cat([cell_patch, tissue_patch], dim=1)

#         cell_prediction = self.model(model_input, (pair_id,)).squeeze(0).detach().cpu()
#         softmaxed = softmax(cell_prediction, dim=0)[:3]
#         result = get_point_predictions(softmaxed)
#         return result


class SegformerJointPred2InputModel(EvaluationModel):

    def __init__(
        self, metadata, cell_model, cell_transform=None, tissue_transform=None
    ):
        super().__init__(metadata, cell_model, None)
        backbone_model = "b3"
        self.cell_transform = cell_transform
        self.tissue_transform = tissue_transform

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

        if self.cell_transform is not None:
            cell_transformed = self.cell_transform(
                image=cell_patch,
                mask=cell_patch,
            )
            cell_patch = cell_transformed["image"]

        if self.tissue_transform is not None:
            tissue_transformed = self.tissue_transform(
                image=tissue_patch,
                mask=tissue_patch,
            )
            tissue_patch = tissue_transformed["image"]

        cell_patch = self._scale_patch(cell_patch)
        tissue_patch = self._scale_patch(tissue_patch, do_scale=True)

        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0).to(self.device)

        tissue_patch = torch.from_numpy(tissue_patch).permute(2, 0, 1)
        tissue_patch = tissue_patch.unsqueeze(0).to(self.device)

        offsets = torch.tensor([[x_offset, y_offset]])

        cell_prediction, _ = self.model(cell_patch, tissue_patch, offsets)
        cell_prediction = cell_prediction.squeeze(0).detach().cpu()
        softmaxed = softmax(cell_prediction, dim=0)

        result = get_point_predictions(softmaxed)
        return result


class SegformerAdditiveJointPred2DecoderModel(SegformerJointPred2InputModel):

    def __init__(
        self, metadata, cell_model, cell_transform=None, tissue_transform=None
    ):
        super().__init__(metadata, cell_model, None)
        backbone_model = "b3"
        self.cell_transform = cell_transform
        self.tissue_transform = tissue_transform
        self.input_image_size = 1024
        self.output_image_size=1024

        if isinstance(cell_model, str):
            self.model = SegformerTissueToCellDecoderModule(
                backbone_model=backbone_model,
                pretrained_dataset="ade",
                input_image_size=self.input_image_size,
                output_image_size=self.output_image_size,
            )
            self.model.load_state_dict(torch.load(cell_model))
        elif isinstance(cell_model, torch.nn.Module):
            self.model = cell_model
        else:
            raise ValueError("Invalid model type ")


import cv2

class SegformerAdditiveJointPred2DecoderWithTTAModel(SegformerJointPred2InputModel):

    def __init__(
        self, metadata, cell_model, cell_transform=None, tissue_transform=None
    ):
        super().__init__(metadata, cell_model, None)
        backbone_model = "b3"
        self.cell_transform = cell_transform
        self.tissue_transform = tissue_transform
        self.input_image_size = 1024
        self.output_image_size=1024

        if isinstance(cell_model, str):
            self.model = SegformerTissueToCellDecoderModule(
                backbone_model=backbone_model,
                pretrained_dataset="ade",
                input_image_size=self.input_image_size,
                output_image_size=self.output_image_size,
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

        if self.cell_transform is not None:
            cell_transformed = self.cell_transform(
                image=cell_patch,
                mask=cell_patch,
            )
            cell_patch = cell_transformed["image"]

        if self.tissue_transform is not None:
            tissue_transformed = self.tissue_transform(
                image=tissue_patch,
                mask=tissue_patch,
            )
            tissue_patch = tissue_transformed["image"]

        cell_patch = self._scale_patch(cell_patch)
        tissue_patch = self._scale_patch(tissue_patch, do_scale=True)

        offset = np.asarray([x_offset, y_offset])

        # TTA
        tissue_patch_tta, offsets = self.geometric_test_time_augmentation(tissue_patch, offset)
        cell_patch_tta, _ = self.geometric_test_time_augmentation(cell_patch, offset)

        results = []

        for tissue_patch, cell_patch, offset in zip(tissue_patch_tta, cell_patch_tta, offsets):
            cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
            cell_patch = cell_patch.unsqueeze(0).to(self.device)

            tissue_patch = torch.from_numpy(tissue_patch).permute(2, 0, 1)
            tissue_patch = tissue_patch.unsqueeze(0).to(self.device)

            offset = torch.from_numpy(offset).unsqueeze(0)

            cell_prediction, _ = self.model(cell_patch, tissue_patch, offset)
            cell_prediction = cell_prediction.squeeze(0).detach().cpu()
            softmaxed = softmax(cell_prediction, dim=0)

            result = softmaxed.numpy()
            results.append(result)

        result = self.reverse_tta(results)
        result = torch.from_numpy(result)
        result = get_point_predictions(result)

        return result

    def geometric_test_time_augmentation(self, img: np.ndarray, offset: np.ndarray) -> List[np.ndarray]:
            """
            Return all 8 possible geometric transformations of an image
            With corresponding offset
            """

            transformed = []
            offsets = []
            for flip in [None, 1]:
                for rotate in [
                    None,
                    cv2.ROTATE_90_CLOCKWISE,
                    cv2.ROTATE_180,
                    cv2.ROTATE_90_COUNTERCLOCKWISE,
                ]:
                    t_img = cv2.flip(img, flip) if flip is not None else img
                    offset = self.pos_flipped_yaxis(offset) if flip is not None else offset
                    t_img = cv2.rotate(t_img, rotate) if rotate is not None else t_img
                    offset = self.pos_rotated_center(position=offset, angle=rotate) if rotate is not None else offset
                    transformed.append(t_img)
                    offsets.append(offset)
            return transformed, offsets

    def reverse_tta(self, pred: np.ndarray) -> np.ndarray:
        """Combine test-time augmentation predictions into a single prediction"""
        i = 0
        pred = torch.Tensor(np.array(pred))
        for flip in [None, 2]:
            for rotate in [None, 1, 2, 3]:
                if rotate:
                    pred[i] = torch.rot90(pred[i], k=rotate, dims=(1, 2))
                if flip is not None:
                    pred[i] = torch.flip(pred[i], dims=[flip])
                i += 1
        mean_pred = torch.mean(pred, dim=0)
        return mean_pred.numpy()

    # Finds the corresponding postion in the image flipped around the y-axis
    def pos_flipped_yaxis(self, position):
        flipped_pos = np.array([1-position[0], position[1]])
        return flipped_pos
    
    def rotate_position_90_counterclock(self, position):
        if position[0] > 0.5 and position[1] > 0.5:
            return np.array([1-position[0], position[1]])
        elif position[0] < 0.5 and position[1] > 0.5:
            return np.array([position[0], 1-position[1]])
        elif position[0] < 0.5 and position[1] < 0.5:
            return np.array([1-position[0], position[1]])
        elif position[0] > 0.5 and position[1] < 0.5:
            return np.array([position[0], 1-position[1]])
    
    # Finds the corresponding position in the image rotated with angle
    def pos_rotated_center(self, position, angle):
        if angle == cv2.ROTATE_90_CLOCKWISE:
            pos = self.rotate_position_90_counterclock(position)
            pos = self.rotate_position_90_counterclock(pos)
            rot_pos = self.rotate_position_90_counterclock(pos)
        elif angle==cv2.ROTATE_180:
            pos = self.rotate_position_90_counterclock(position)
            rot_pos = self.rotate_position_90_counterclock(pos)
        elif angle==cv2.ROTATE_90_COUNTERCLOCKWISE:
            rot_pos = self.rotate_position_90_counterclock(position)
        return rot_pos

        
        
