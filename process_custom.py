import os
import sys
import json
import albumentations as A

from pathlib import Path
from util import gcio

sys.path.append(os.getcwd())

# from user.inference import Deeplabv3CellOnlyModel as Model

from user.inference import Deeplabv3TissueCellModel as Model

# from user.inference import SegFormerCellOnlyModel as Model

# from user.inference import Deeplabv3TissueFromFile as Model

from src.utils.constants import IDUN_OCELOT_DATA_PATH


def process_model_output():
    """Process a test patches. This involves iterating over samples,
    inferring and write the cell predictions
    """
    # Set this to "val" or "test"
    partition = "val"
    tissue_from_file = False

    offsets = {
        "train": 0,
        "val": 400,
        "test": 537,
    }

    cell_model_path = "outputs/models/20240314_163849_deeplabv3plus-cell-branch_pretrained-1_lr-1e-04_dropout-0.3_backbone-resnet50_normalization-macenko_id-1_epochs-60.pth"
    tissue_model_path = "outputs/models/best/20240313_002829_deeplabv3plus-tissue-branch_pretrained-1_lr-1e-04_dropout-0.1_backbone-resnet50_normalization-macenko_id-5_best.pth"
    # tissue_model_path = None
    cell_file_path = os.path.join(
        IDUN_OCELOT_DATA_PATH, f"images/{partition}/cell_macenko/"
    )

    # Cell detection writer
    prediction_output_path = Path(
        f"{os.getcwd()}/eval_outputs/cell_classification_{partition}.json"
    )

    metadata_path = os.path.join(IDUN_OCELOT_DATA_PATH, "metadata.json")
    meta_dataset = gcio.read_json(metadata_path)
    meta_dataset = list(meta_dataset["sample_pairs"].values())[offsets[partition] :]

    # 'mask' means don't apply transform, 'image' means do apply
    additional_targets = {
        "image": "mask",
        "tissue": "mask",
    }
    model = Model(meta_dataset, cell_model_path, tissue_model_path)

    if tissue_from_file:
        tissue_path = os.path.join(
            IDUN_OCELOT_DATA_PATH, f"annotations/{partition}/predicted_cropped_tissue/"
        )
    else:
        tissue_path = os.path.join(
            IDUN_OCELOT_DATA_PATH, f"images/{partition}/tissue_macenko/"
        )

    transforms = A.Compose([A.Normalize()], additional_targets=additional_targets)
    loader = gcio.CustomDataLoader(cell_file_path, tissue_path)

    pred_data = {
        "type": "Multiple points",
        "points": [],
        "version": {"major": 1, "minor": 0},
    }
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
        # writer.add_points(cell_classification, pair_id)
        for x, y, c, prob in cell_classification:
            pred_data["points"].append(
                {
                    "name": f"image_{str(pair_id)}",
                    "point": [int(x), int(y), int(c)],
                    "probability": prob,
                }
            )

    # Export the prediction into a json file
    # writer.save()
    with open(prediction_output_path, "w", encoding="utf-8") as f:
        json.dump(pred_data, f, ensure_ascii=False, indent=4)
    print(f"Saved predictions to {prediction_output_path}")


if __name__ == "__main__":
    process_model_output()
