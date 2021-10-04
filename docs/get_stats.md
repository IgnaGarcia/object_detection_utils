# GET STATS
Script to evaluate object detection model and export stats

## Explain
This scripts run the exported model and detect object in image inputs, for earch image compare with expected detections and get the IoU of evary detection, with that we can filter the 'valid' and 'invalid' detections, and later calc statics to resume our model in some numbers and export into csv file

## Command Template
Note that score and iou are opcionals, the default values are:
- score = 0.1
- iou = '0.1,0.3,0.5,0.7,0.9'
```
python get_stats.py \
    --l {path_to_labels} \
    --m {path_to_exported_model} \
    --inp {path_to_images_and_xml} \
    --out {path_to_csv_ouutput} \
    [--score {min_score_to_valid_detection} \]
    [--iou {iou_list_to_filter}]
```

## Command Example
```
python get_stats.py \
    --l annotations/label_map.pbtxt \
    --m exported-models/my_model/ \
    --inp images/test \
    --out images/output \
    --score 0.7 \
    --iou 0.3,0.6,0.9
```

## Output Example
| image | iou | expected | expectedByClass | totalDetected | totalDetectedByClass | validDetected | validDetectedByClass | q1validConfidence | q2validConfidence | q3validConfidence | q1invalidConfidence | q2invalidConfidence | q3invalidConfidence | precision | precisionByClass | recall | recallByClass | f1 | f1ByClass |  
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| img1 | 0.4 | 28 | 14-14 | 29 | 14-15 | 28 | 14-14 | 0.957 | 0.986 | 0.992 | 0.67 | 0.67 | 0.67 | 0.966 | 1.0-0.933 | 1 | 1.0-1.0 | 0.983 | 1.0-0.965 |
| img1 | 0.8 | 28 | 14-14 | 29 | 14-15 | 28 | 14-14 | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. |
| img2 | 0.4 | 18 | 4-14 | 20 | 5-15 | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. |
| img2 | 0.8 | 18 | 4-14 | 20 | 5-15 | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. |
| img3 | 0.4 | 15 | 10-5 | 19 | 14-5 | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. |
| img3 | 0.8 | 15 | 10-5 | 19 | 14-5 | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. | .. |