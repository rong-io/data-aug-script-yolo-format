from bs4 import BeautifulSoup
import imgaug as ia
import imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
from tqdm import trange
from lxml import etree
import glob
from pathlib import Path
import os
import argparse

def convert_to_yolo(label, xmin, ymin, xmax, ymax, img_width, img_height):
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return f"{label} {x_center} {y_center} {width} {height}"

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

parser = argparse.ArgumentParser(description='Augment images and labels for YOLO.')
parser.add_argument('--label_dir', type=str, required=True, help='Path to the label directory')
parser.add_argument('--image_dir', type=str, required=True, help='Path to the image directory')
parser.add_argument('--aug_label_dir', type=str, required=True, help='Path to the augmented label directory')
parser.add_argument('--aug_image_dir', type=str, required=True, help='Path to the augmented image directory')
parser.add_argument('--multiplier', type=int, required=True, help='Number of times to augment each image')
args = parser.parse_args()

seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 水平翻轉50%的圖像
        iaa.Affine(rotate=(-20, 20)),  # 隨機旋轉-20到20度
        iaa.GaussianBlur(sigma=(0, 3.0)),  # 應用0到3.0的高斯模糊
        iaa.Multiply((0.8, 1.2)),  # 隨機增亮或減暗圖像
        iaa.LinearContrast((0.75, 1.5)),  # 增強或減弱對比度
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # 添加高斯噪音
        iaa.Crop(percent=(0, 0.2)),  # 隨機裁剪0%到20%的圖像
    ])

for img_path in glob.glob(args.image_dir + "*.jpg"):
    img_name = Path(img_path).stem

    with open(args.label_dir + img_name + ".txt", "r") as f:  # 原標註檔
        lines = f.readlines()
    
    image = imageio.imread(args.image_dir + img_name + ".jpg")  # 原影像，須注意副檔名
    bbsOnImg = []
    
    for line in lines:
        label, xc, yc, w, h = map(float, line.strip().split())
        x1 = max(0, (xc - w / 2) * image.shape[1])
        x2 = min(image.shape[1], (xc + w / 2) * image.shape[1])
        y1 = max(0, (yc - h / 2) * image.shape[0])
        y2 = min(image.shape[0], (yc + h / 2) * image.shape[0])
        bbsOnImg.append(BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2, label=str(int(label))))
    
    bbs = BoundingBoxesOnImage(bbsOnImg, shape=image.shape)

    for j in range(args.multiplier):
        j = j + 1
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        bbs_aug_clip = bbs_aug.clip_out_of_image()
        
        foldername = args.aug_label_dir
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

        aug_img_path = args.aug_image_dir
        if not os.path.exists(aug_img_path):
            os.makedirs(aug_img_path)
        
        imageio.imsave(aug_img_path + filename, image_aug)