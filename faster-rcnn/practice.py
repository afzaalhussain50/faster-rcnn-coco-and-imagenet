import os
import json
detections_path = '/home/seecs/afzaalhussain/thesis/fvqa-data/new_dataset_release/scg-detections'
images_path = '/home/seecs/afzaalhussain/thesis/fvqa-data/new_dataset_release/images'
detections_file_path = [os.path.join(detections_path, x) for x in os.listdir(detections_path)]
detection_file = detections_file_path[0]
f = open(detection_file)
detections_json = json.load(f)
print(detection_file)
print(detections_json['filename'])
# del detections_json['coco-faster-rcnn']
detections_json.pop('coco-faster-rcnn', None)
print(detections_json)
coco_faster_rnn_results = {'index' : [1, 3], 'labels' : ['person', 'cake']}
detections_json['coco-faster-rcnn'] = coco_faster_rnn_results
print(detections_json['coco-faster-rcnn'])
with open(detection_file, 'w') as f:
    json.dump(detections_json, f)
    print('done', detection_file)

