import torch
import sys
import numpy as np

sys.path.append("/cluster/work/jssaethe/histopathology_segmentation")
from src.deeplabv3.network.modeling import _segm_resnet
from torch.nn.functional import softmax
from skimage.feature import peak_local_max


class Model:
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
            num_channels=3,
            output_stride=8,
            pretrained_backbone=pretrained_backbone,
            dropout_rate=dropout_rate,
        )
        self.model.load_state_dict(
            torch.load(
                "outputs/models/2024-01-20_19-50-43_deeplabv3plus_cell_only_lr-1e-05_dropout-0.3_backbone-resnet50_epochs-100.pth"
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
