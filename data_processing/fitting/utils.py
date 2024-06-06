import cv2
import numpy as np
import torch
import sys
def read_exr(filename: str) -> np.ndarray:
    """
    Read exr image

    Args:
        filename: image filename

    Returns:
        image: a ndarray of shape (H, W, C). The three channels are in 
            RBG format.
    """
    assert(filename.endswith(".exr"))
    image = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if len(image.shape) == 3:
        image = image[:, :, [2, 1, 0]]
    return image


def read_ldr(filename: str) -> np.array:
    """
    Returns:
        image: RGB format
    """
    image = cv2.imread(filename)
    if image is None:
        return image
    image = image[:, :, ::-1]
    return image


def write_rgb_image(filename: str, image: np.array):
    """
    OpenCV!!
    
    Args:
        image: RGB image (H, W, 3)
    """
    image = image.astype(np.float32)

    if len(image.shape) == 3:
        assert(image.shape[2] in [1, 3])
        # convert RGB to BRG
        image = image[:, :, ::-1]
    
    cv2.imwrite(filename, image)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_multiprocess():
    if sys.platform == 'linux':
        torch.multiprocessing.set_start_method('spawn', force=True)