"""Reading images and listing dataset directories."""

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(directory, suffixes=IMAGE_SUFFIXES):
    """Return the image paths under `directory`, sorted.

    TODO: implement.
    """
    raise NotImplementedError


def load_image(path, mode="RGB"):
    """Load a single image as a numpy array.

    TODO: implement.
    """
    raise NotImplementedError


def load_images(paths, mode="RGB"):
    """Load several images as a list of numpy arrays.

    TODO: implement.
    """
    raise NotImplementedError
