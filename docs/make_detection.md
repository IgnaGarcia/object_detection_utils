# MAKE DETECTION
Script to run our exported model and get images with detection boxes

## Explain
This scripts run the exported model and detect object in image inputs, for earch image generate a new image with detection boxes and save into output dir, also generate txt file with sumarry of this run, wich contains count of detectections for each image and each class

## Command Template
Note that score and ckpt are opcional, the default values are:
- score = 0.7
- ckpt = 0
```
python make_detection.py \
    -m {path_to_exported_model} \
    --inp {path_to_images} \
    --out {path_to_images_output} \
    --l {path_to_labelmap} \
    [--score {min_score_to_valid_detection} \]
    [--ckpt {steps_of_last_checkpoint}] 
```

## Command Example
```
python make_detection.py \
    -m exported-models/my_model/ \
    --inp images/detect \
    --out images/output \
    --l annotations/label_map.pbtxt \
    --score 0.7 \
    --ckpt 1500 
```

## Output Example
```
6 imagenes
2 clases - (white-black)
NOMBRE IMAGEN->TOTAL-clase1-clase2-...-claseN
detect (1).jpg->20-9-11
detect (2).jpg->8-4-4
detect (3).jpg->16-9-7
detect (4).jpg->8-3-5
detect (5).jpg->9-5-4
detect (6).jpg->9-4-5
TOTAL->70
TOTAL POR CLASE->34-36
PROBABILIDAD->0.3
CHECKPOINT->7500
MODELO->test2clases_faster_inception
IOU_THRESHOLD->0.6000000238418579

```
![detect](./detect(1).png)