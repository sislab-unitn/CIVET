mkdir -p cocoapi/images

# val data
wget http://images.cocodataset.org/zips/val2017.zip -P data/coco/images
unzip data/coco/images/val2017.zip -d cocoapi/images

# test data
wget http://images.cocodataset.org/zips/test2017.zip -P data/coco/images
unzip data/coco/images/test2017.zip -d cocoapi/images

# annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P data/coco/annotations
wget http://images.cocodataset.org/annotations/image_info_test2017.zip -P data/coco/annotations
unzip data/coco/annotations/annotations_trainval2017.zip -d cocoapi
unzip data/coco/annotations/image_info_test2017.zip -d cocoapi