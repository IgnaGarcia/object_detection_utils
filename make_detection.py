'''
Script to detect objects from set of images and export images with his boxes

command run example:

    python make_detection.py -m exported-models/my_faster_rcnn_resnet101_640x640_coco/ --inp images/detect --out images/output --score 0.7 --ckpt 500 --l annotations/label_map.pbtxt
'''

import os
import pathlib
import tensorflow as tf
import time
from datetime import datetime 
import numpy as np

import matplotlib
from PIL import Image
from absl import app, flags

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

## Definir FLAGS
flags.DEFINE_string('m', None, 'Path to model')
flags.DEFINE_string('l', None, 'Path to label map')
flags.DEFINE_string('inp', None, 'Path to images to test')
flags.DEFINE_string('out', None, 'Path to images results')
flags.DEFINE_float('score', 0.7, 'Min score')
flags.DEFINE_integer('ckpt', 0, 'Checkpoint step')

flags.mark_flag_as_required('m')
flags.mark_flag_as_required('l')
flags.mark_flag_as_required('inp')
flags.mark_flag_as_required('out')

FLAGS = flags.FLAGS


## Funcion para cargar el Modelo
def load_model(model_path):
    global iouThreshold
    PATH_TO_CFG = model_path + "/pipeline.config"
    PATH_TO_CKPT = model_path + "/checkpoint"

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    
    finds = (str(model_config)).split(" iou_threshold: ")
    if (len(finds) >= 2):
    	iouThreshold = finds[1].split(" ")[0]
    else:
    	iouThreshold = "null"
    	
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

    return detection_model

## Realizar la deteccion
def detect_fn(model, image):
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections

## Recibe un array, lo recorre y lo devuelve separado por guiones
def arrToStr(arr):
    str = ""
    for i, e in enumerate(arr):
        if(i == len(arr)-1):
            str += f'{e}'
        else:
            str += f'{e}-'
    return str
    
## Recibe los objetos de label map y retorna un array de los nombres
def getClases(categories):
    clases = []
    for i in categories:
        clases.append(categories[i]['name'])
    return clases

## Guardar en archivo la cantidad de objetos
def save_count(count, imageName):
    global today 
    global modelName
    # count
    with open(f"{_get_img_output()}/{modelName} {today} count.txt", "a") as f:
        f.write(f'{imageName}->{count}\n')
        f.close()
    # count summary
    with open(f"{_get_img_output()}/{modelName} summary_count.txt", "a") as f:
        f.write(f'{imageName}->{count}\n')
        f.close()


def innitWrite(TEST_IMAGE_PATHS, category):
    global today 
    global modelName
    # count
    f = open(f"{_get_img_output()}/{modelName} {today} count.txt", "w") # si existe el archivo lo vacia
    f.write(f'{len(TEST_IMAGE_PATHS)} imagenes\n')
    f.write(f'{len(category)} clases - ({arrToStr(getClases(category))})\n')
    f.write(f'NOMBRE IMAGEN->TOTAL-clase1-clase2-...-claseN\n')
    f.close()
    
    # count summary
    f = open(f"{_get_img_output()}/{modelName} summary_count.txt", "a")
    f.write(f'\n--------------------------\n')
    f.write(f'{today}\n')
    f.write(f'{len(TEST_IMAGE_PATHS)} imagenes\n')
    f.write(f'{len(category)} clases - ({arrToStr(getClases(category))})\n')
    f.write(f'NOMBRE IMAGEN->TOTAL-clase1-clase2-...-claseN\n')
    f.close()

## Contar cantidad de objetos detectados
def count_objects(scores, minScore, classes, category_index):
    global GLOBAL_COUNT 
    global GLOBAL_COUNT_PER_CLASS
    
    indexes = [k for k,v in enumerate(scores) if (v > minScore)]
    GLOBAL_COUNT += len(indexes)
    
    count_per_class = []
    for i in range(0,len(category_index)):
        count_per_class.append(0)
    
    for idx, el in enumerate(classes):
        if(scores[idx] > minScore):
            GLOBAL_COUNT_PER_CLASS[el] += 1
            count_per_class[el] += 1
            #print(f"\n\t\tCLASS {el} - {category_index[el+1]['name']}")
    
    print(f'\tNUMERO DE OBJETOS {len(indexes)}')
    print(f'\tNUMERO DE OBJETOS POR CLASE {arrToStr(count_per_class)}')
    return f'{len(indexes)}-{arrToStr(count_per_class)}'

## Hacer prediccion para una imagen
def run_inference_for_single_image(model, image):
    input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
    output_dict = detect_fn(model, input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        
    return output_dict

## Mostrar la prediccion
def show_inference(model, image_path, category_index):
    global today 
    global modelName
    image_np = np.array(Image.open(image_path))
    minScore = _get_score()
    detections = run_inference_for_single_image(model, image_np)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    vis_util.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=minScore,
            agnostic_mode=False)
    
    # contar objetos y guardarlos en txt
    count = count_objects(detections['detection_scores'], minScore, detections['detection_classes'], category_index)
    save_count(count, image_path.parts[-1])

    # guardar
    print(f'\tGUARDANDO en {_get_img_output()}/{modelName} {today} {image_path.parts[-1][0:-4]}.png')
    matplotlib.image.imsave(f'{_get_img_output()}/{modelName} {today} {image_path.parts[-1][0:-4]}.png', image_np_with_detections)

## Flag Wrappers
def _get_label():
    return pathlib.Path(FLAGS.l)

def _get_model():
    return FLAGS.m

def _get_img_input():
    return pathlib.Path(FLAGS.inp)

def _get_img_output():
    return FLAGS.out

def _get_score():
    return FLAGS.score

def _get_ckpt():
    return FLAGS.ckpt

## Flujo principal
def main(argv):
    global GLOBAL_COUNT 
    global GLOBAL_COUNT_PER_CLASS 
    global today
    global iouThreshold
    global modelName
    GLOBAL_COUNT = 0
    GLOBAL_COUNT_PER_CLASS = []
    modelName = pathlib.Path(_get_model()).parts[-1]
    today = datetime.now().strftime("%Y-%b-%d_%H:%M")

    start_time = time.time()
    print(f'\n\tINFO\n\t\tLABELMAP:\t{_get_label()}\n\t\tMODEL:\t{_get_model()}\n\t\tINPUT:\t{_get_img_input()}\n\t\tOUTPUT:\t{_get_img_output()}\n\t\tMIN SCORE:\t{_get_score()}')
    
    ## Cargar el Label Map
    category_index = label_map_util.create_category_index_from_labelmap(_get_label(), use_display_name=True)
    for i in range(0,len(category_index)):
        GLOBAL_COUNT_PER_CLASS.append(0)
        
    ## Cargar Imagenes de entrada
    TEST_IMAGE_PATHS = sorted(list(_get_img_input().glob("*.jpg")) + \
                                    list(_get_img_input().glob("*.JPG")) + \
                                    list(_get_img_input().glob("*.PNG")) + \
                                    list(_get_img_input().glob("*.png")))
    ## Cargar Modelo
    detection_model = load_model(_get_model())

    innitWrite(TEST_IMAGE_PATHS, category_index)

    for image_path in TEST_IMAGE_PATHS:
        img_time = time.time()
        print(f'\n\tINICIO\tIMAGE {image_path}')
        show_inference(detection_model, image_path, category_index)
        print(f'\tFIN\tIMAGE {image_path} - {round(time.time()-img_time, 2)}segs')
    
    save_count(GLOBAL_COUNT, "TOTAL")
    save_count(arrToStr(GLOBAL_COUNT_PER_CLASS), "TOTAL POR CLASE")
    save_count(_get_score(), "PROBABILIDAD")
    save_count(_get_ckpt(), "CHECKPOINT")
    save_count(modelName, "MODELO")
    save_count(iouThreshold, "IOU_THRESHOLD")
    
    print(f'\n\tFIN\tEl programa tardo {round(time.time()-start_time, 2)}segs')
    print(f'\tTOTAL\t{GLOBAL_COUNT} objetos detectados')
    print(f'\tTOTAL POR CLASE\t{arrToStr(GLOBAL_COUNT_PER_CLASS)} objetos detectados')

if __name__ == '__main__':
    app.run(main)
