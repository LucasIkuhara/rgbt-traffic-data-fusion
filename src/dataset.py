from pycocotools import coco
import skimage.io as io
import matplotlib.pyplot as plt

DATASET_PATH = "aau-rainsnow"
rgbAnnFile = DATASET_PATH + "/aauRainSnow-rgb.json"
thermalAnnFile = DATASET_PATH + "/aauRainSnow-thermal.json"

rainSnowRgbGt = coco.COCO(rgbAnnFile)
rainSnowThermalGt = coco.COCO(thermalAnnFile)

for i in range(0, 2197):
    chosenImgId = i
    annIds = rainSnowRgbGt.getAnnIds(imgIds=[chosenImgId])
    anns = rainSnowRgbGt.loadAnns(annIds)

    rgbImg = rainSnowRgbGt.loadImgs([chosenImgId])[0]
    thermalImg = rainSnowThermalGt.loadImgs([chosenImgId])[0]
    thermalAnns = rainSnowThermalGt.loadAnns(annIds)

    print(
        "Found "
        + str(len(anns))
        + " annotations at image ID "
        + str(chosenImgId)
        + ". Image file: "
        + rgbImg["file_name"]
    )

    for ann in anns:
        print(
            f'Annotation #{ann["id"]}: '
            + rainSnowRgbGt.loadCats(ann["category_id"])[0]["name"]
        )
        print(ann)

    print("\nRGB Image")
    I = io.imread(f"{DATASET_PATH}/" + rgbImg["file_name"])
    plt.gcf().clear()
    plt.imshow(I)
    rainSnowRgbGt.showAnns(anns)
    plt.show()
    break
    # For some reason, the image won't show in some Windows/Anaconda configurations. If this is the case, print the image instead
    # plt.savefig("Samples/rgb-" + str(chosenImgId).zfill(5) + ".png")
