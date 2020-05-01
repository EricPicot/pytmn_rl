import keyboard as kb
import cv2
import numpy as np
from PIL import ImageGrab
import tf2_processing
import tensorflow as tf

up = [1, 0, 0, 0, 0, 0, 0, 0, 0]
down = [0, 1, 0, 0, 0, 0, 0, 0, 0]
right = [0, 0, 1, 0, 0, 0, 0, 0, 0]
left = [0, 0, 0, 1, 0, 0, 0, 0, 0]
up_right = [0, 0, 0, 0, 1, 0, 0, 0, 0]
up_left = [0, 0, 0, 0, 0, 1, 0, 0, 0]
down_right = [0, 0, 0, 0, 0, 0, 1, 0, 0]
down_left = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nothing = [0, 0, 0, 0, 0, 0, 0, 0, 1]


def grab_screen(region, speed_region):
    y_min, x_min, y_max, x_max = speed_region
    image = np.array(ImageGrab.grab(bbox=region))
    speed = image[x_min: x_max, y_min: y_max]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.cvtColor(speed, cv2.COLOR_BGR2RGB)

def key_check():
    key_list = []
    key = kb.get_hotkey_name()
    if key:
        key_list.append(key)
        key_list = list(set(key_list))
    return key_list

def prediction_to_keys(prediction):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    if prediction.argmax() == 0:
        output = "up"
    elif prediction.argmax() == 1:
        output = "down"
    elif prediction.argmax() == 2:
        output = "right"
    elif prediction.argmax() == 3:
        output = "left"
    elif prediction.argmax() == 4:
        output = "right+up"
    elif prediction.argmax() == 5:
        output = "left+up"
    else:
        output = "up"

    print(prediction, output)
    return output


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    if 'up' in keys:
        output = up
    elif 'down' in keys:
        output = down
    elif 'right' in keys:
        output = right
    elif 'left' in keys:
        output = left
    elif 'right+down' in keys:
        output = down_right
    elif 'right+up' in keys:
        output = up_right
    elif 'left+down' in keys:
        output = down_left
    elif 'left+up' in keys:
        output = up_left
    else:
        output = nothing
    return output

def transform_target(target):
    target_list = []
    for t in target:
        transformed_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        transformed_target[t] = 1
        target_list.append(transformed_target)
    return np.array(target_list).reshape((len(target_list), 10))

def speed_numerisation(image, model):
    """
    takes a speed image and return its value in integer
    """
    image = np.asarray(image).astype(np.float32) / 255

    first_digit, second_digit, third_digit = tf2_processing.digit_images(image)
    first = np.argmax(model.predict(tf.reshape(first_digit, (1, 30, 20, 1)),  steps=1))
    second = np.argmax(model.predict(tf.reshape(second_digit, (1, 30, 20, 1)),  steps=1))
    third = np.argmax(model.predict(tf.reshape(third_digit, (1, 30, 20, 1)),  steps=1))
    return first * 100 + second * 10 + third