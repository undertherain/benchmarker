from pathlib import Path

import cv2
from pycocotools.coco import COCO

from helpers import get_file


def get_data(params, N=20):
    BASE = Path("~/.cache/benchmarker/").expanduser()
    COCODIR = BASE.joinpath("data/coco/")
    coco = COCO(COCODIR.joinpath("annotations/instances_val2017.json"))
    imgIds = coco.getImgIds()[:N]
    imgStructs = coco.loadImgs(imgIds)
    # each element has keys: ['license', 'file_name', 'coco_url',
    # 'height', 'width', 'date_captured', 'flickr_url', 'id']

    def _get_file(imgStruct):
        from os.path import basename

        imgUrl = imgStruct["coco_url"]
        fname = basename(imgUrl)
        return get_file(fname, imgUrl, cache_subdir="images", cache_dir=COCODIR)

    file_paths = map(_get_file, imgStructs)
    X = list(map(cv2.imread, file_paths))
    return X


if __name__ == "__main__":
    for arr in get_data(None):
        print(arr.shape)
