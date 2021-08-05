import os
import json
import cv2

underwater_classes = ['holothurian', 'echinus', 'scallop', 'starfish']

root_dir = '/media/chiyukunpeng/CHENPENG01/contest/underwater_object_detection20200827/data/test-A-image/'
images = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
label_file = "/media/chiyukunpeng/CHENPENG01/contest/underwater_object_detection20200827/results/cas_x101.bbox.json"
test_json = json.load(open(label_file, 'r'))
raw_label_file = "/media/chiyukunpeng/CHENPENG01/contest/underwater_object_detection20200827/data/annotations/test-A.json"
test_json_raw = json.load(open(raw_label_file, "r"))

# convert image_id to image_filename
imgid2name = {}
for imageinfo in test_json_raw['images']:
    imgid = imageinfo['id']
    imgid2name[imgid] = imageinfo['file_name']
for anno in test_json:
    img_id = anno['image_id']
    filename = imgid2name[img_id]
    anno['image_id'] = filename

for image in images:
    img = cv2.imread(image)
    image = str(os.path.basename(image))
    for anno in test_json:
        if anno['image_id'] == image:
            xmin, ymin, xmax, ymax = int(anno['bbox'][0]),int(anno['bbox'][1]),int(anno['bbox'][2]),int(anno['bbox'][3])
            confidence = round(anno['score'], 2)
            class_id = int(anno['category_id'])
            class_name = underwater_classes[class_id-1]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(img, str(confidence), (xmin, ymin - 20), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)
            cv2.putText(img, str(class_name), (xmin+20, ymin - 20), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)
    cv2.imshow('img', img)
    cv2.waitKey(0)

