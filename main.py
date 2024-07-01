import imgaug as ia
import imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
from tqdm import trange
from lxml import etree
import glob
from pathlib import Path
import os


def convert_to_yolo(label, xmin, ymin, xmax, ymax, img_width, img_height):
    # 計算物件中心的相對座標和寬度/高度
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return f"{label} {x_center} {y_center} {width} {height}"


# Create functions
class CreateAnnotations:
    def __init__(self, foldername, filename):
        self.root = etree.Element("annotation")
        child1 = etree.SubElement(self.root, "folder")
        child1.text = foldername

        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename

        child3 = etree.SubElement(self.root, "path")
        child3.text = filename

        child4 = etree.SubElement(self.root, "source")

        child5 = etree.SubElement(child4, "database")
        child5.text = "Unknown"

    def set_size(self, imgshape):
        (height, width, channel) = imgshape
        self.img_width = width
        self.img_height = height

    def savefile(self, filename, yolo_labels):
        with open(filename, "w") as f:
            for label in yolo_labels:
                f.write(label + "\n")

    def add_pic_attr(self, label, xmin, ymin, xmax, ymax):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)


# 定義路徑變數
label_dir = ""
image_dir = ""
aug_label_dir = "./result/labels/"
aug_image_dir = "./result/images/"

# 擴充張數設定
seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # 左右翻轉
        iaa.Affine(rotate=(-10, 10)),  # 旋轉
        iaa.Multiply((0.8, 1.2)),  # 改變亮度
    ]
)

for img_path in glob.glob(image_dir + "*.jpg"):
    img_name = Path(img_path).stem

    with open(label_dir + img_name + ".txt", "r") as f:  # 原標註檔
        lines = f.readlines()

    image = imageio.imread(image_dir + img_name + ".jpg")  # 原影像，須注意副檔名
    bbsOnImg = []

    for line in lines:
        label, xc, yc, w, h = map(float, line.strip().split())
        x1 = max(0, (xc - w / 2) * image.shape[1])
        x2 = min(image.shape[1], (xc + w / 2) * image.shape[1])
        y1 = max(0, (yc - h / 2) * image.shape[0])
        y2 = min(image.shape[0], (yc + h / 2) * image.shape[0])
        bbsOnImg.append(BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2, label=str(int(label))))

    bbs = BoundingBoxesOnImage(bbsOnImg, shape=image.shape)

    for j in range(200):
        j = j + 1
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        bbs_aug_clip = bbs_aug.clip_out_of_image()

        foldername = aug_label_dir
        filename = img_name + "_" + str(j) + ".jpg"

        anno = CreateAnnotations(foldername, filename)
        anno.set_size(image_aug.shape)
        yolo_labels = []

        if not os.path.exists(foldername):
            os.makedirs(foldername)

        for index, bb in enumerate(bbs_aug_clip):
            xmin = int(bb.x1)
            ymin = int(bb.y1)
            xmax = int(bb.x2)
            ymax = int(bb.y2)
            label = str(bb.label)
            anno.add_pic_attr(label, xmin, ymin, xmax, ymax)
            yolo_label = convert_to_yolo(
                label, xmin, ymin, xmax, ymax, anno.img_width, anno.img_height
            )
            yolo_labels.append(yolo_label)

        anno.savefile("{}{}.txt".format(foldername, filename[:-4]), yolo_labels)

        aug_img_path = aug_image_dir
        if not os.path.exists(aug_img_path):
            os.makedirs(aug_img_path)

        imageio.imsave(aug_img_path + filename, image_aug)
