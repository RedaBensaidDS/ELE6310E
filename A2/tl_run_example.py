# Usage: python3 tl_run_example.py

import os
import pytimeloop.timeloopfe.v4 as tl

THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    start_dir = os.path.join(THIS_SCRIPT_DIR, "Q1")
    out_dir = os.path.join(start_dir, "output")
    spec = tl.Specification.from_yaml_files(
        os.path.join(start_dir, "arch/*.yaml"), # Note: We expect * to resolve to only one file
        os.path.join(start_dir, "map/Q1_os-tiled.map.yaml"),
        os.path.join(start_dir, "prob/*.yaml"),
    )
    
    tl.call_model(spec, output_dir=os.path.join(start_dir, f"{out_dir}"))
