'''
Script to evaluate object detection model and export stats

command run example:
    python get_stats.py \
        --l annotations/label_map.pbtxt \
        --m exported-models/my_faster_rcnn_resnet101_640x640_coco/ \
        --inp images/test \
        --out images/output \
        --score 0.7 \
        --iou 0.3,0.6,0.9
'''
## Define IMPORTS
import os
import pathlib
import collections
import time
import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from absl import app, flags
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder


## Define FLAGS
flags.DEFINE_string('l', None, 'Path to label map')
flags.DEFINE_string('m', None, 'Path to model')
flags.DEFINE_string('inp', None, 'Path to pair xml-images to evaluate')
flags.DEFINE_string('out', None, 'Path to exported csv, default root of model')
flags.DEFINE_float('score', 0.1, 'Min score to make detections')
flags.DEFINE_list('iou', '0.1,0.3,0.5,0.7,0.9', 'IoU to evaluate')

flags.mark_flag_as_required('m')
flags.mark_flag_as_required('l')
flags.mark_flag_as_required('inp')

FLAGS = flags.FLAGS

def getLabelPath():
    return pathlib.Path(FLAGS.l)

def getModelPath():
    return FLAGS.m

def getInputPath():
    return pathlib.Path(FLAGS.inp)

def getOutputPath():
    if(FLAGS.out == None):
        return FLAGS.m
    return FLAGS.out

def getScore():
    return FLAGS.score

def getIoU():
    return [float(i) for i in FLAGS.iou]

def log(tag, message):
    print(f'\n\t--{tag.upper()}: {message}')


## Loads
# Load Model
def loadModel():
    configPath = getModelPath() + '/pipeline.config'
    checkpointPath = getModelPath() + '/checkpoint'

    configs = config_util.get_configs_from_pipeline_file(configPath)
    modelConfig = configs['model']
    	
    detectionModel = model_builder.build(model_config=modelConfig, is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detectionModel)
    ckpt.restore(os.path.join(checkpointPath, 'ckpt-0')).expect_partial()

    return detectionModel

# Load Images
def loadImages():
    return sorted(list(getInputPath().glob('*.jpg')) + \
                    list(getInputPath().glob('*.JPG')) + \
                    list(getInputPath().glob('*.PNG')) + \
                    list(getInputPath().glob('*.png')))

# Load Labelmap
def loadLabelmap():
    return label_map_util.create_category_index_from_labelmap(getLabelPath(), use_display_name=True)

# Load Expected Images
def loadExpected():
    expecteds = {}
    for xml_file in getInputPath().glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        objects = []
        for member in root.findall('object'):
            row = {
                'class': member[0].text,
                'coords': {
                    'ymin': int(member[4][1].text),
                    'ymax': int(member[4][3].text),
                    'xmin': int(member[4][0].text),
                    'xmax': int(member[4][2].text)
                }
            }
            objects.append(row)
        expecteds[xml_file.stem] = objects
    return expecteds


## Detect an create objects with data to calc stats
# Get real coordinates from boxes
def getCoordinate(image, boxes):
    height, width, channels = image.shape
    ymin, xmin, ymax, xmax = boxes.tolist()
    
    return {
        'ymin': int(ymin * height), 
        'ymax': int(ymax * height), 
        'xmin': int(xmin * width), 
        'xmax': int(xmax * width)
    }

# Filter detections with confidence > minScore and create object to calc stats
def filterDetections(detections, score, categoryIndex, image):   
    count = [0, []]
    filtered = []

    for i in range(0, len(categoryIndex)):
        count[1].append(0)
    
    for idx, el in enumerate(detections['detection_classes']):
        if(detections['detection_scores'][idx] > score):
            count[1][el] += 1
            count[0] += 1
            filtered.append( {
                'coords': getCoordinate(image, detections['detection_boxes'][idx]), 
                'class': categoryIndex[el+1]['name'], 
                'score': detections['detection_scores'][idx]
            } )

    return {'count': count, 'detections': filtered}

# Make detection
def runInference(model, image):
    inputTensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)

    inputTensor, shapes = model.preprocess(inputTensor) 
    predictionDict = model.predict(inputTensor, shapes)
    outputDict = model.postprocess(predictionDict, shapes)

    numDetections = int(outputDict.pop('num_detections'))
    outputDict = {key:value[0, :numDetections].numpy() 
                    for key,value in outputDict.items()}
    outputDict['num_detections'] = numDetections
    outputDict['detection_classes'] = outputDict['detection_classes'].astype(np.int64)
        
    return outputDict

# Open image with np, run detection, coun objects and return stats
def calcInference(model, imagePath, categoryIndex):
    imageNp = np.array(Image.open(imagePath))

    return filterDetections(
        runInference(model, imageNp), 
        getScore(), 
        categoryIndex, 
        imageNp.copy()
    )


## Calc Statistics
# Calc IoU
def calcIoU(boxA, boxB):
    # Intersection Boxes
	yMin = max(boxA['ymin'], boxB['ymin'])
	yMax = min(boxA['ymax'], boxB['ymax'])
	xMin = max(boxA['xmin'], boxB['xmin'])
	xMax = min(boxA['xmax'], boxB['xmax'])

	# Area of Intersection
	interArea = max(0, xMax - xMin + 1) * max(0, yMax - yMin + 1)

	# Area of each Boxes
	boxAArea = (boxA['ymax'] - boxA['ymin'] + 1) * (boxA['xmax'] - boxA['xmin'] + 1)
	boxBArea = (boxB['ymax'] - boxB['ymin'] + 1) * (boxB['xmax'] - boxB['xmin'] + 1)

	# Areas - Interesection Area
	return interArea / float(boxAArea + boxBArea - interArea)

# Search the max IoU for each image
def addMaxIoU(detections, detectExpecteds):  
    for detect in detections:
        maxIoU = 0
        classMaxIoU = None

        for expected in detectExpecteds:
          iou = calcIoU(detect['coords'], expected['coords'])
          if(maxIoU < iou):
              maxIoU = iou
              classMaxIoU = expected['class']

        detect['iou'] = maxIoU

        if classMaxIoU == detect['class']:
          detect['trueClass'] = True
        else:
          detect['trueClass'] = False

    return detections

# Calc Average Precision
def calcMAvgPrecision():
    # TODO: calc
    return 0

# Calc F1 [ 2 ( precision * recall / (precision + recall) )  ]
def calcF1(precision, recall):
    if precision + recall == 0:
        return 0
    return round(2 * ( (precision * recall) / (precision + recall) ), 3)

# Calc Recall [  TP / (TP + FN)  ]
def calcRecall(tp, fn):
    if tp + fn == 0:
        return 0
    return round(tp / ( tp + fn ), 3)

# Calc Precision [  TP / (TP + FP)  ]
def calcPrecision(tp, fp):
    if tp + fp == 0:
        return 0
    return round(tp / ( tp + fp ), 3)

# Iterate for calc Precision & Recall
def summaryStats(detected, valid, expected):
    precision = [0, []]
    recall = [0, []]
    f1 = [0, []]

    falsePositive = detected[0] - valid[0]
    falseNegative = expected[0] - valid[0]

    precision[0] = calcPrecision(valid[0], falsePositive)
    recall[0] = calcRecall(valid[0], falseNegative)
    f1[0] = calcF1(precision[0], recall[0])

    for i, el in enumerate(detected[1]):
        falsePositive = detected[1][i] - valid[1][i]
        falseNegative = expected[1][i] - valid[1][i]

        precision[1].append(calcPrecision(valid[1][i], falsePositive))
        recall[1].append(calcRecall(valid[1][i], falseNegative))
        f1[1].append(calcF1(precision[1][i], recall[1][i]))

    return [precision, recall, f1]

# Calc Average Confidence
def calcConfidence(detection):
    acc = 0
    for detect in detection['detections']:
        acc += detect['score']
    return round(acc / detection['count']['total'], 3)

# Get all statics
def getStats(detection, iou, countExpect, categoryIndex):
    valid = [0, []]
    validConfidences = []
    invalidConfidences = []
    classes = {}
    
    for el in categoryIndex.values():
        valid[1].append(0)
        classes[el['name']] = el['id']
    
    for el in detection['detections']:
        if el['iou'] >= iou:
            valid[0] += 1
            validConfidences.append(el['score'])
            if el['trueClass']:
                valid[1][classes[el['class']]-1] += 1
        else:
            invalidConfidences.append(el['score'])
    
    qValids, qInvalids = [[0, 0, 0], [0, 0, 0]]
    if validConfidences:
      qValids = np.percentile(validConfidences, [25, 50, 75])
    if invalidConfidences:
      qInvalids = np.percentile(invalidConfidences, [25, 50, 75])

    precision, recall, f1 = summaryStats(detection['count'], valid, countExpect)

    stats = {
        'valid': valid[0],
        'validByClass': valid[1],

        'q1ValidConfidence': round(qValids[0], 3),
        'q2ValidConfidence': round(qValids[1], 3),
        'q3ValidConfidence': round(qValids[2], 3),

        'q1InvalidConfidence': round(qInvalids[0], 3),
        'q2InvalidConfidence': round(qInvalids[1], 3),
        'q3InvalidConfidence': round(qInvalids[2], 3),

        'precision': precision[0], 
        'precisionByClass': precision[1],

        'recall': recall[0],
        'recallByClass': recall[1], 

        'f1': f1[0],
        'f1ByClass': f1[1]
        #'mAPrecision': calcMAvgPrecision()
    }
    return stats

def getExpectedsCount(expected, categoryIndex):
    count = [0, []]
    classes = {}
    
    for el in categoryIndex.values():
        count[1].append(0)
        classes[el['name']] = el['id']
    
    for idx, el in enumerate(expected):
        count[0] += 1
        count[1][classes[el['class']]-1] += 1

    return count

## Export stats with format:
# Write row in output file
def write(fileName, row):    
    f = open(fileName, 'a')
    f.write(f'{row}\n')
    f.close()

# Create file to write latter
def createFile(fileName):
    '''
    create file with columns:
      -image

      -expected
      -expectedByClass

      -totalDetected
      -totalDetectedByClass
      -validDetected
      -validDetectedByClassIoU

      -q1validConfidence
      -q2validConfidence
      -q3validConfidence

      -q1invalidConfidence
      -q2invalidConfidence
      -q3invalidConfidence
      
      -precision
      -precisionByClass

      -recall
      -recallByClass

      -f1
      -f1ByClass

      mAvgPrecision
    '''
    f = open(fileName, 'w')
    f.write(f'image,iou,expected,expectedByClass,totalDetected,totalDetectedByClass,validDetected,validDetectedByClass,q1validConfidence,q2validConfidence,q3validConfidence,q1invalidConfidence,q2invalidConfidence,q3invalidConfidence,precision,precisionByClass,recall,recallByClass,f1,f1ByClass\n')
    f.close()

# Create row to write
def getRow(img, iou, expect, detect, stats):
    expectXClass = arrToStr(expect[1])
    detectXClass = arrToStr(detect["count"][1])
    validXClass = arrToStr(stats["validByClass"])
    precisionXClass = arrToStr(stats["precisionByClass"])
    recallXClass = arrToStr(stats["recallByClass"])
    f1ByClass = arrToStr(stats["f1ByClass"])

    expectRow = f'{expect[0]},{expectXClass}'
    detectRow = f'{detect["count"][0]},{detectXClass}'         
    validRow = f'{stats["valid"]},{validXClass}'
    validConfRow = f'{stats["q1ValidConfidence"]},{stats["q2ValidConfidence"]},{stats["q3ValidConfidence"]}'
    invalidConfRow = f'{stats["q1InvalidConfidence"]},{stats["q2InvalidConfidence"]},{stats["q3InvalidConfidence"]}'       
    precisionRow = f'{stats["precision"]},{precisionXClass}'
    recallRow = f'{stats["recall"]},{recallXClass}'
    f1Row = f'{stats["f1"]},{f1ByClass}'

    return f'{img.stem},{iou},{expectRow},{detectRow},{validRow},{validConfRow},{invalidConfRow},{precisionRow},{recallRow},{f1Row}'


## Main
def arrToStr(arr):
    str = ""
    for i, e in enumerate(arr):
        if(i == len(arr)-1):
            str += f'{e}'
        else:
            str += f'{e}-'
    return str

def main(argv):
    today = datetime.now().strftime('%Y-%b-%d_%H:%M')
    modelName = pathlib.Path(getModelPath()).parts[-1]  
    fileName = f'{getOutputPath()}/stats_{modelName}_{today}.csv'

    log("start", "loading resources")
    categoryIndex = loadLabelmap()
    images = loadImages()
    model = loadModel()
    expecteds = loadExpected()

    createFile(fileName)

    for image in images:
        if(image.stem in expecteds):
            log(image.stem, "processing")
            countExpect = getExpectedsCount(expecteds[image.stem], categoryIndex)
            detections = calcInference(model, image, categoryIndex)
            detections['detections'] = addMaxIoU(detections['detections'], expecteds[image.stem])

            for iou in getIoU():
              stats = getStats(detections, iou, countExpect, categoryIndex)
              row = getRow(image, iou, countExpect, detections, stats)
              log(f"iou {iou}", row)
              write(fileName, row)
        else:
            log(image.stem, "image whitout xml pair")

    log("end", fileName)


if __name__ == '__main__':
    app.run(main)