# -*- encoding=utf-8 -*-
import os
import random

xmlFilePath = r'./Annotations'
BasePath = r'./ImageSets/Main'

trainval_percent=1
train_percent=1
def main():
    temp_xml = os.listdir(xmlFilePath)
    total_xml = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)
    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)
    ftrainval = open(os.path.join(BasePath,'trainval.txt'), 'w')
    ftest = open(os.path.join(BasePath,'test.txt'), 'w')
    ftrain = open(os.path.join(BasePath,'train.txt'), 'w')
    fval = open(os.path.join(BasePath,'val.txt'), 'w')

    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
    ftrainval.close()
    ftrain.close()
    ftest.close()
    fval.close()


if __name__ == "__main__":
    main()