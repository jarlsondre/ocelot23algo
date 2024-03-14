import os
import sys
import albumentations as A

from pathlib import Path
from util import gcio

sys.path.append(os.getcwd())

from user.inference import Deeplabv3CellOnlyModel as Model

# from user.inference import Deeplabv3TissueCellModel as Model

# from user.inference import SegFormerCellOnlyModel as Model

from user.inference import Deeplabv3TissueLeakingModel

from src.utils.constants import CELL_IMAGE_MEAN, CELL_IMAGE_STD

DATA_DIR = "/cluster/projects/vc/data/mic/open/OCELOT/ocelot_data"


def process_model_output():
    """Process a test patches. This involves iterating over samples,
    inferring and write the cell predictions
    """
    # Set this to "val" or "test"
    partition = "val"
    tissue_leaking = False

    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    cell_path = os.path.join(DATA_DIR, f"images/{partition}/cell_macenko/")
    output_path = Path(
        f"{os.getcwd()}/eval_outputs/cell_classification_{partition}.json"
    )

    # Cell detection writer
    writer = gcio.DetectionWriter(output_path)

    # Loading metadata
    meta_dataset = gcio.read_json(metadata_path)
    meta_dataset = list(meta_dataset["sample_pairs"].values())

    # Getting the correct file locations depending on the model
    if tissue_leaking:
        additional_targets = {
            "image": "mask",
            "tissue": "mask",  # Don't apply normalization to tissue_patch
        }
        tissue_ending = ".png"
        tissue_path = os.path.join(
            DATA_DIR, f"annotations/{partition}/predicted_cropped_tissue/"
        )
        model = Deeplabv3TissueLeakingModel(meta_dataset)
    else:
        additional_targets = {
            "image": "image",
            "tissue": "image",
        }
        tissue_ending = ".jpg"
        tissue_path = os.path.join(DATA_DIR, f"images/{partition}/tissue_macenko/")
        model = Model(meta_dataset)

    # Creating transforms and loader
    transforms = A.Compose(
        [A.Normalize()],
        additional_targets=additional_targets,
    )
    loader = gcio.CustomDataLoader(
        cell_path,
        tissue_path,
        tissue_ending=tissue_ending,
    )

    # NOTE: Batch size is 1
    for cell_patch, tissue_patch, pair_id in loader:
        print(f"Processing sample pair {pair_id}")
        # Cell-tissue patch pair inference
        cell_classification = model(
            cell_patch,
            tissue_patch,
            pair_id,
            transform=transforms,
        )

        # Updating predictions
        writer.add_points(cell_classification, pair_id)

    # Export the prediction into a json file
    writer.save()


if __name__ == "__main__":
    process_model_output()
