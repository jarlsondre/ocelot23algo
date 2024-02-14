from pathlib import Path

# TODO: Remove "test" in front of paths

# Grand Challenge folders were input files can be found
GC_CELL_FPATH = Path("test/input/images/cell_patches/")
GC_TISSUE_FPATH = Path("test/input/images/tissue_patches/")

GC_METADATA_FPATH = Path("test/input/metadata.json")

# Grand Challenge output file
GC_DETECTION_OUTPUT_PATH = Path("test/output/cell_classification.json")

# Sample dimensions
SAMPLE_SHAPE = (1024, 1024, 3)
