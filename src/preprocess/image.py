import os
from pathlib import Path

vips_bin = Path(r"C:\Program Files\vips-dev-8.17\bin")
if os.name == "nt" and vips_bin.is_dir():
    os.environ["PATH"] = str(vips_bin) + os.pathsep + os.environ.get("PATH", "")

import pyvips


def open_image(path):
    return pyvips.Image.new_from_file(path, access="random")