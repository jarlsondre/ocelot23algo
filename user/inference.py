import os
import sys
import torch
import numpy as np
import torch.nn.functional as F

from torch.nn.functional import softmax, interpolate

sys.path.append(os.getcwd())

from src.models import DeepLabV3plusModel
from src.utils.utils import crop_and_resize_tissue_patch, get_point_predictions
from src.utils.constants import CELL_IMAGE_MEAN, CELL_IMAGE_STD

from skimage.feature import peak_local_max
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig,
    SegformerImageProcessor,
)


def validate_inputs(cell_patch: np.ndarray, tissue_patch: np.ndarray):
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

    def __init__(self, metadata):
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
        self.model.load_state_dict(
            torch.load(
                # Normalization off
                # "outputs/models/20240312_030603_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-off_id-1_best.pth"
                # "outputs/models/20240312_030603_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-off_id-2_best.pth"
                # "outputs/models/20240312_030609_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-off_id-3_best.pth"
                # "outputs/models/20240312_031531_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-off_id-4_best.pth"
                # "outputs/models/20240312_031531_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-off_id-5_best.pth"
                # ImageNet Normalization
                # "outputs/models/20240312_031531_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-imagenet_id-1_best.pth"
                # "outputs/models/20240312_031531_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-imagenet_id-2_best.pth"
                # "outputs/models/20240312_031531_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-imagenet_id-3_best.pth"
                # "outputs/models/20240312_031530_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-imagenet_id-4_best.pth"
                # "outputs/models/20240312_074706_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-imagenet_id-5_best.pth"
                # Cell Normalization
                # "outputs/models/20240312_094820_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-cell_id-1_best.pth"
                # "outputs/models/20240312_143840_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-cell_id-2_best.pth"
                # "outputs/models/20240312_150435_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-cell_id-3_best.pth"
                # "outputs/models/20240312_164425_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-cell_id-4_best.pth"
                # "outputs/models/20240312_165437_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-cell_id-5_best.pth"
                # Macenko Normalization
                # "outputs/models/20240312_181739_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko_id-1_best.pth"
                # "outputs/models/20240312_184334_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko_id-2_best.pth"
                # "outputs/models/20240312_185942_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko_id-3_best.pth"
                # "outputs/models/20240312_190612_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko_id-4_best.pth"
                # "outputs/models/20240312_205029_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko_id-5_best.pth"
                # Macenko + Cell
                # "outputs/models/20240312_205029_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko + cell_id-1_best.pth"
                # "outputs/models/20240312_205029_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko + cell_id-2_best.pth"
                # "outputs/models/20240312_205029_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko + cell_id-3_best.pth"
                # "outputs/models/20240312_210030_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko + cell_id-4_best.pth"
                # "outputs/models/20240312_212404_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko + cell_id-5_best.pth"
                # Macenko + ImageNet
                # "outputs/models/20240312_212404_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko + imagenet_id-1_best.pth"
                # "outputs/models/20240312_214124_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko + imagenet_id-2_best.pth"
                # "outputs/models/20240312_215531_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko + imagenet_id-3_best.pth"
                # "outputs/models/20240312_215531_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko + imagenet_id-4_best.pth"
                "outputs/models/20240312_215531_deeplabv3plus-cell-only_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko + imagenet_id-5_best.pth"
            )
        )
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, cell_patch, tissue_patch, pair_id, transform=None):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided.

        NOTE: this implementation offers a dummy inference example. This must be
        updated by the participant.

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

        #############################################
        #### YOUR INFERENCE ALGORITHM GOES HERE #####
        #############################################
        validate_inputs(cell_patch, tissue_patch)

        if transform:
            transformed = transform(image=cell_patch)
            cell_patch = transformed["image"]

        # Setting range to [0, 1] if not already done
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

    def __init__(self, metadata):
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
        self.tissue_branch.load_state_dict(
            torch.load(
                # "outputs/models/20240303_205501_deeplabv3plus-tissue-branch_pretrained-1_lr-6e-05_dropout-0.1_backbone-resnet50_epochs-100.pth"
                # "outputs/models/best/20240313_002829_deeplabv3plus-tissue-branch_pretrained-1_lr-1e-04_dropout-0.1_backbone-resnet50_normalization-macenko_id-5_best.pth"
                "outputs/models/best/20240313_014914_deeplabv3plus-tissue-branch_pretrained-1_lr-1e-04_dropout-0.1_backbone-resnet50_normalization-macenko + imagenet_id-3_best.pth"
            )
        )
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
        self.cell_branch.load_state_dict(
            torch.load(
                # "outputs/models/20240228_200511_deeplabv3plus-tissue-leaking_pretrained-1_lr-5e-05_dropout-0.3_backbone-resnet50_epochs-100.pth"
                # "outputs/models/20240228_192603_deeplabv3plus-cell-branch_pretrained-1_lr-5e-05_dropout-0.3_backbone-resnet50_epochs-100.pth"
                # "outputs/models/20240305_162509_deeplabv3plus-cell-branch_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_epochs-100.pth"
                "outputs/models/best/20240314_090119_deeplabv3plus-cell-branch_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko_and_imagenet_id-1_best.pth"
            )
        )
        self.cell_branch.eval()
        self.cell_branch.to(self.device)

    def __call__(self, cell_patch, tissue_patch, pair_id, transform=None):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided.

        NOTE: this implementation offers a dummy inference example. This must be
        updated by the participant.

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

        #############################################
        #### YOUR INFERENCE ALGORITHM GOES HERE #####
        #############################################
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

        if tissue_patch.dtype == np.uint8:
            tissue_patch = tissue_patch.astype(np.float32) / 255.0
        elif tissue_patch.dtype != np.float32:
            tissue_patch = tissue_patch.astype(np.float32)

        # Preparing shape and device for model input
        tissue_patch = torch.from_numpy(tissue_patch).permute(2, 0, 1)
        tissue_patch = tissue_patch.unsqueeze(0).to(self.device)

        # Predicting tissue
        tissue_prediction = self.tissue_branch(tissue_patch).squeeze(0).detach().cpu()
        tissue_prediction = tissue_prediction.argmax(dim=0)

        # Getting metadata to crop data
        tissue_mpp = meta_pair["tissue"]["resized_mpp_x"]
        cell_mpp = meta_pair["cell"]["resized_mpp_x"]
        x_offset = meta_pair["patch_x_offset"]
        y_offset = meta_pair["patch_y_offset"]

        cropped_tissue: torch.Tensor = crop_and_resize_tissue_patch(
            image=tissue_prediction,
            tissue_mpp=tissue_mpp,
            cell_mpp=cell_mpp,
            x_offset=x_offset,
            y_offset=y_offset,
        )

        tissue_prediction = F.one_hot(cropped_tissue, num_classes=3).permute(2, 0, 1)
        tissue_prediction = tissue_prediction.unsqueeze(0)

        # Cell Patch
        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0)

        # Concatenating to create final input
        model_input = torch.cat([cell_patch, tissue_prediction], dim=1)
        model_input = model_input.to(self.device)

        # Getting prediction
        cell_prediction = self.cell_branch(model_input).squeeze(0).detach().cpu()
        softmaxed = softmax(cell_prediction, dim=0)
        result = get_point_predictions(softmaxed)
        return result


class Deeplabv3TissueLeakingModel:
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """

    def __init__(self, metadata):
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
        self.model.load_state_dict(
            torch.load(
                # "outputs/models/20240228_192603_deeplabv3plus-cell-branch_pretrained-1_lr-5e-05_dropout-0.3_backbone-resnet50_epochs-100.pth"
                # "outputs/models/20240228_200511_deeplabv3plus-tissue-leaking_pretrained-1_lr-5e-05_dropout-0.3_backbone-resnet50_epochs-100.pth"
                # "outputs/models/best/20240314_090119_deeplabv3plus-cell-branch_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko_and_imagenet_id-1_best.pth"
                # "outputs/models/best/20240314_163849_deeplabv3plus-cell-branch_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko_id-1_best.pth"
                "outputs/models/20240314_163849_deeplabv3plus-cell-branch_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko_id-1_epochs-60.pth"
            )
        )
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, cell_patch, tissue_patch, pair_id, transform=None) -> list:
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided.

        NOTE: this implementation offers a dummy inference example. This must be
        updated by the participant.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8]
            Tissue patch with shape [1024, 1024, 3] with values from 0 or 1
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        # Getting the metadata corresponding to the patch pair ID
        meta_pair = self.metadata[pair_id]

        #############################################
        #### YOUR INFERENCE ALGORITHM GOES HERE #####
        #############################################
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

        # tissue_patch = tissue_patch.astype(np.float32)
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

    def __init__(self, metadata):
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
        self.model.load_state_dict(
            torch.load(
                "outputs/models/20240223_203530_segformer-cell-only_pretrained-1_lr-1e-04_epochs-100.pth"
            )
        )
        self.image_processor = SegformerImageProcessor(
            do_resize=False, do_normalize=True
        )
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, cell_patch, tissue_patch, pair_id):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided.

        NOTE: this implementation offers a dummy inference example. This must be
        updated by the participant.

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

        #############################################
        #### YOUR INFERENCE ALGORITHM GOES HERE #####
        #############################################

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
