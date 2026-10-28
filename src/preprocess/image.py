import pyvips


def open_image(path):
    return pyvips.Image.new_from_file(path, access="random")