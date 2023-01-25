from typing import Optional, Tuple

import cv2


def read_image(path: str, size: Optional[Tuple[int, int]] = None):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size is not None:
        image = cv2.resize(image, size)
    return image
