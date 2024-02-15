import os
import json
import numpy as np

from pathlib import Path
from util import gcio
from util.constants import (
    GC_DETECTION_OUTPUT_PATH,
)

# from user.inference import Deeplabv3CellOnlyModel as Model
from user.inference import Deeplabv3TissueCellModel as Model

# from user.inference import SegFormerCellOnlyModel as Model

DATA_DIR = "/cluster/projects/vc/data/mic/open/OCELOT/ocelot_data"


def process_model_output():
    """Process a test patches. This involves iterating over samples,
    inferring and write the cell predictions
    """

    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    cell_path = os.path.join(DATA_DIR, "images/val/cell/")
    tissue_path = os.path.join(DATA_DIR, "images/val/tissue/")
    output_path = Path(
        "/cluster/work/jssaethe/histopathology_segmentation/ocelot23algo/test/output/cell_classification.json"
    )

    # Initialize the data loader
    loader = gcio.CustomDataLoader(cell_path, tissue_path)

    # Cell detection writer
    writer = gcio.DetectionWriter(output_path)

    # Loading metadata
    meta_dataset = gcio.read_json(metadata_path)
    meta_dataset = list(meta_dataset["sample_pairs"].values())

    # Instantiate the inferring model
    model = Model(meta_dataset)

    # NOTE: Batch size is 1
    for cell_patch, tissue_patch, pair_id in loader:
        print(f"Processing sample pair {pair_id}")
        # Cell-tissue patch pair inference
        cell_classification = model(cell_patch, tissue_patch, pair_id)

        # Updating predictions
        writer.add_points(cell_classification, pair_id)

    # Export the prediction into a json file
    writer.save()


if __name__ == "__main__":
    process_model_output()
