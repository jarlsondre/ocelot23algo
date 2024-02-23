import os
import sys
import torch
import numpy as np

sys.path.append(os.getcwd())
from src.deeplabv3.network.modeling import _segm_resnet
from src.models import DeepLabV3plusModel
from src.utils.utils import crop_and_upscale_tissue
from torch.nn.functional import softmax, interpolate
from skimage.feature import peak_local_max
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig,
    SegformerImageProcessor,
)


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

        # self.model = _segm_resnet(
        #     name="deeplabv3plus",
        #     backbone_name=backbone_model,
        #     num_classes=3,
        #     num_channels=3,
        #     output_stride=8,
        #     pretrained_backbone=pretrained_backbone,
        #     dropout_rate=dropout_rate,
        # )
        self.model = DeepLabV3plusModel(
            backbone_name=backbone_model,
            num_classes=3,
            num_channels=3,
            pretrained=pretrained_backbone,
            dropout_rate=dropout_rate,
        )
        self.model.load_state_dict(
            torch.load(
                "outputs/models/20240223_111913_deeplabv3plus-cell-only_pretrained-True_lr-1e-04_dropout-0.3_backbone-resnet50_epochs-20.pth"
            )
        )
        self.model.eval()
        self.model.to(self.device)

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

        cell_patch = (
            torch.tensor(cell_patch).permute(2, 0, 1).to(torch.float32).unsqueeze(0)
        )
        cell_patch = cell_patch.to(self.device)
        cell_patch = cell_patch / 255.0

        output = self.model(cell_patch).squeeze(0).detach().cpu()
        softmaxed = softmax(output, dim=0)

        # max values and indices
        confidences, predictions = torch.max(softmaxed, dim=0)
        confidences, predictions = confidences.numpy(), predictions.numpy()
        peak_points_pred = peak_local_max(
            confidences,
            min_distance=20,
            labels=np.logical_or(predictions == 1, predictions == 2),
            threshold_abs=0.01,
        )
        xs = []
        ys = []
        probs = []
        ids = []
        for x, y in peak_points_pred:
            probability = confidences[x, y]
            class_id = predictions[x, y]
            if class_id == 2:
                class_id = 1
            elif class_id == 1:
                class_id = 2
            xs.append(y.item())
            ys.append(x.item())
            probs.append(probability.item())
            ids.append(class_id)

        #############################################
        ####### RETURN RESULS PER SAMPLE ############
        #############################################

        # We need to return a list of tuples with 4 elements, i.e.:
        # - int: cell's x-coordinate in the cell patch
        # - int: cell's y-coordinate in the cell patch
        # - int: class id of the cell, either 1 (BC) or 2 (TC)
        # - float: confidence score of the predicted cell
        return list(zip(xs, ys, ids, probs))


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
        self.tissue_branch = _segm_resnet(
            name="deeplabv3plus",
            backbone_name=backbone_model,
            num_classes=3,
            num_channels=3,
            output_stride=8,
            pretrained_backbone=pretrained_backbone,
            dropout_rate=dropout_rate,
        )
        self.tissue_branch.load_state_dict(
            torch.load(
                "outputs/models/2024-01-21_15-48-32_deeplabv3plus_tissue_branch_lr-1e-05_dropout-0.3_backbone-resnet50_epochs-30.pth"
            )
        )
        self.tissue_branch.eval()
        self.tissue_branch.to(self.device)

        # Create cell branch
        self.cell_branch = _segm_resnet(
            name="deeplabv3plus",
            backbone_name=backbone_model,
            num_classes=3,
            num_channels=6,
            output_stride=8,
            pretrained_backbone=pretrained_backbone,
            dropout_rate=dropout_rate,
        )
        self.cell_branch.load_state_dict(
            torch.load(
                "outputs/models/20240218_001933_deeplabv3plus_cell_branch_lr-0.0001_dropout-0.3_backbone-resnet50_epochs-100.pth"
            )
        )
        self.cell_branch.eval()
        self.cell_branch.to(self.device)

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
        tissue_patch = (
            torch.tensor(tissue_patch).permute(2, 0, 1).to(torch.float32).unsqueeze(0)
        )
        tissue_patch = tissue_patch.to(self.device)
        tissue_patch = tissue_patch / 255.0
        tissue_prediction = self.tissue_branch(tissue_patch).squeeze(0).detach().cpu()

        # Cropping the tissue image and upscaling it to the original size
        offset_tensor = (
            torch.tensor([meta_pair["patch_y_offset"], meta_pair["patch_x_offset"]])
            * 1024
        )

        # x and y mpp are the same
        scaling_value = (
            meta_pair["cell"]["resized_mpp_x"] / meta_pair["tissue"]["resized_mpp_x"]
        )
        cropped_tissue = crop_and_upscale_tissue(
            tissue_tensor=tissue_prediction,
            offset_tensor=offset_tensor,
            scaling_value=scaling_value,
        )

        # Creating one-hot for the tissue prediction
        argmaxed = torch.argmax(cropped_tissue, dim=0)
        mask = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        tissue_prediction = mask[argmaxed].permute(2, 0, 1)

        cell_patch = (
            torch.tensor(cell_patch).permute(2, 0, 1).to(torch.float32).unsqueeze(0)
        )
        cell_patch = cell_patch / 255.0

        model_input = torch.cat([cell_patch, tissue_prediction.unsqueeze(0)], dim=1)
        model_input = model_input.to(self.device)

        cell_prediction = self.cell_branch(model_input).squeeze(0).detach().cpu()
        softmaxed = softmax(cell_prediction, dim=0)

        # max values and indices
        confidences, predictions = torch.max(softmaxed, dim=0)
        confidences, predictions = confidences.numpy(), predictions.numpy()
        peak_points_pred = peak_local_max(
            confidences,
            min_distance=20,
            labels=np.logical_or(predictions == 1, predictions == 2),
            threshold_abs=0.01,
        )
        xs = []
        ys = []
        probs = []
        ids = []
        for x, y in peak_points_pred:
            probability = confidences[x, y]
            class_id = predictions[x, y]
            if class_id == 2:
                class_id = 1
            elif class_id == 1:
                class_id = 2
            xs.append(y.item())
            ys.append(x.item())
            probs.append(probability.item())
            ids.append(class_id)

        #############################################
        ####### RETURN RESULS PER SAMPLE ############
        #############################################

        # We need to return a list of tuples with 4 elements, i.e.:
        # - int: cell's x-coordinate in the cell patch
        # - int: cell's y-coordinate in the cell patch
        # - int: class id of the cell, either 1 (BC) or 2 (TC)
        # - float: confidence score of the predicted cell
        return list(zip(xs, ys, ids, probs))


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

        self.model = _segm_resnet(
            name="deeplabv3plus",
            backbone_name=backbone_model,
            num_classes=3,
            num_channels=6,
            output_stride=8,
            pretrained_backbone=pretrained_backbone,
            dropout_rate=dropout_rate,
        )
        self.model.load_state_dict(
            torch.load(
                "outputs/models/20240218_002202_deeplabv3plus_tissue_leaking_lr-0.0001_dropout-0.3_backbone-resnet50_epochs-100.pth"
            )
        )
        self.model.eval()
        self.model.to(self.device)

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

        cell_patch = cell_patch.astype(np.float32) / 255.0
        cell_patch = torch.from_numpy(cell_patch).permute(2, 0, 1)
        cell_patch = cell_patch.unsqueeze(0)

        tissue_patch = tissue_patch.astype(np.float32)
        tissue_patch = torch.from_numpy(tissue_patch).permute(2, 0, 1)
        tissue_patch = tissue_patch.unsqueeze(0)

        model_input = torch.cat([cell_patch, tissue_patch], dim=1)
        model_input = model_input.to(self.device)

        cell_prediction = self.model(model_input).squeeze(0).detach().cpu()
        softmaxed = softmax(cell_prediction, dim=0)

        # max values and indices
        confidences, predictions = torch.max(softmaxed, dim=0)
        confidences, predictions = confidences.numpy(), predictions.numpy()
        peak_points_pred = peak_local_max(
            confidences,
            min_distance=20,
            labels=np.logical_or(predictions == 1, predictions == 2),
            threshold_abs=0.01,
        )
        xs = []
        ys = []
        probs = []
        ids = []
        for x, y in peak_points_pred:
            probability = confidences[x, y]
            class_id = predictions[x, y]
            if class_id == 2:
                class_id = 1
            elif class_id == 1:
                class_id = 2
            xs.append(y.item())
            ys.append(x.item())
            probs.append(probability.item())
            ids.append(class_id)

        #############################################
        ####### RETURN RESULS PER SAMPLE ############
        #############################################

        # We need to return a list of tuples with 4 elements, i.e.:
        # - int: cell's x-coordinate in the cell patch
        # - int: cell's y-coordinate in the cell patch
        # - int: class id of the cell, either 1 (BC) or 2 (TC)
        # - float: confidence score of the predicted cell
        return list(zip(xs, ys, ids, probs))


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
                "outputs/models/20240218_001834_segformer_cell_only_pretrained-0_lr-0.0001_epochs-100.pth"
            )
        )
        self.image_processor = SegformerImageProcessor(do_resize=False)
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

        # max values and indices
        confidences, predictions = torch.max(softmaxed, dim=0)
        confidences, predictions = confidences.numpy(), predictions.numpy()
        peak_points_pred = peak_local_max(
            confidences,
            min_distance=20,
            labels=np.logical_or(predictions == 1, predictions == 2),
            threshold_abs=0.01,
        )
        xs = []
        ys = []
        probs = []
        ids = []
        for x, y in peak_points_pred:
            probability = confidences[x, y]
            class_id = predictions[x, y]
            if class_id == 2:
                class_id = 1
            elif class_id == 1:
                class_id = 2
            xs.append(y.item())
            ys.append(x.item())
            probs.append(probability.item())
            ids.append(class_id)

        #############################################
        ####### RETURN RESULS PER SAMPLE ############
        #############################################

        # We need to return a list of tuples with 4 elements, i.e.:
        # - int: cell's x-coordinate in the cell patch
        # - int: cell's y-coordinate in the cell patch
        # - int: class id of the cell, either 1 (BC) or 2 (TC)
        # - float: confidence score of the predicted cell
        return list(zip(xs, ys, ids, probs))
