from pycocotools import coco
from pycocotools.cocoeval import COCOeval

# DATASET_DIR = Path("dataset")


# def evaluate():
#     results = {}

#     print("=== RGB model on RGB images ===")


#     rgb_model = get_rgb_detector()
#     # results["rgb"] = rgb_model.val(data=str(DATASET_DIR / "rgb.yaml"), split="val")
#     # rgb_res = rgb_model.val(data="dataset/rgb.yaml")
#     # print(
#     #     rgb_res,
#     # )
#     # print("\n=== Thermal model on thermal images ===")
#     # thermal_model = get_thermal_detector()
#     # results["thermal"] = thermal_model.val(data=str(DATASET_DIR / "thermal.yaml"), split="val")

#     # print("\n=== Summary ===")
#     # for name, r in results.items():
#     #     print(f"[{name}] mAP50: {r.box.map50:.4f}  mAP50-95: {r.box.map:.4f}")

#     # return results


# if __name__ == "__main__":
#     evaluate()

DATASET_PATH = "aau-rainsnow/"
RGB_COCO = "aauRainSnow-rgb.json"
rgbAnnFile = DATASET_PATH + "aauRainSnow-rgb.json"


rainSnowRgbGt = coco.COCO(rgbAnnFile)
res = rainSnowRgbGt.loadRes("out.json")

ev = COCOeval(rainSnowRgbGt, res, "bbox")
ev.evaluate()
ev.accumulate()
ev.summarize()

# print(ev)
# print(ev.stats)
print(ev.stats[0])
