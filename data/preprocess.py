import os
import cv2
import numpy as np
from tqdm import tqdm
data_dir = './VOC2012/VOC2012/VOCdevkit/VOC2012'
def preprocess_voc2012(data_dir=data_dir, output_dir='data/datasets/VOC2012/processed'):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, 'JPEGImages')
    labels_dir = os.path.join(data_dir, 'SegmentationClass')

    for img_name in tqdm(os.listdir(images_dir)):
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.png'))

        if os.path.exists(label_path):
            img = cv2.imread(img_path)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            img_output_path = os.path.join(output_dir, 'images')
            label_output_path = os.path.join(output_dir, 'labels')

            os.makedirs(img_output_path, exist_ok=True)
            os.makedirs(label_output_path, exist_ok=True)

            cv2.imwrite(os.path.join(img_output_path, img_name), img)
            cv2.imwrite(os.path.join(label_output_path, img_name.replace('.jpg', '.png')), label)

def main():
    # 预处理 VOC2012 数据集
    preprocess_voc2012()

if __name__ == '__main__':
    main()