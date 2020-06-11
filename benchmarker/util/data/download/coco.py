from pathlib import Path

from pycocotools.coco import COCO


def get_data(params, N=20):
    BASE = Path("~/.cache/benchmarker/").expanduser()
    COCODIR = BASE.joinpath("data/coco/annotations/instances_val2017.json")
    coco = COCO(COCODIR)
    imgIds = coco.getImgIds()[:N]
    imgStructs = coco.loadImgs(imgIds)
    # each element has keys: ['license', 'file_name', 'coco_url',
    # 'height', 'width', 'date_captured', 'flickr_url', 'id']
    imgUrls = map(lambda struct: struct["coco_url"], imgStructs)
    return list(imgUrls)


if __name__ == "__main__":
    print(get_data(None))
