# -*- encoding=utf-8 -*-
import pathlib
import os
import shutil
import cv2

root = pathlib.Path(".")
cw = os.getcwd()

def move() ->None :
    all_img = root.glob("*.png")
    all_xml = root.glob("*.xml")
    all_img = [str(img) for img in all_img]
    all_xml = [str(xm) for xm in all_xml]
    pendix_img = [item.split(".")[0] for item in all_img]
    pendix_xml = [item.split(".")[0] for item in all_xml]
    pendix_img = set(pendix_img)
    pendix_xml = set(pendix_xml)
    union = pendix_img & pendix_xml
    img_list, xml_list = [], []
    os.mkdir("Image")
    os.mkdir("annotation")
    for item in union:
        currentImgPath = os.path.join(cw, item + ".png")
        currentXmlPath = os.path.join(cw, item + ".xml")
        if currentImgPath.endswith(".png"):
            new_item = item + ".png"
            new_path = os.path.join(cw, "Image")
            new_path = os.path.join(new_path, new_item)
            shutil.move(currentImgPath, new_path)
        elif currentXmlPath.endswith(".xml"):
            new_item = item + ".xml"
            new_path = os.path.join(cw, "annotation")
            new_path = os.path.join(new_path, new_item)
            shutil.move(currentXmlPath, new_path)
    return

def changeChannel():
    imgPath = cw + "\\Image"
    ImageRoot = pathlib.Path(imgPath)
    all_img = ImageRoot.glob("*.png")
    all_img = [str(item) for item in all_img]
    error_list = []
    for item in all_img:
        try:
            im = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
            im_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        except IOError:
            print(item)
            error_list.append(item)
        else:
            newName = item.split("\\")[-1]
            newName = newName.split(".")[0] + ".jpg"
            newName = os.path.join(imgPath, newName)
            cv2.imwrite(newName, im_rgb)


if __name__ == "__main__":
    move()
