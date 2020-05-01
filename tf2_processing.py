import random
import numpy as np
import tensorflow as tf


def read_image(path):
    tf_content = np.load(path, allow_pickle=True)
    tf_content = np.asarray(tf_content).astype(np.float32) / 255
    return tf_content


def resize_and_scale_gray(tf_image, resize, augmentation=False, apply_sobel_bool=True, ratio=.4):

    tf_image = tf.image.resize(tf_image, resize)
    tf_image = tf.reshape(tf_image, (-1, resize[0], resize[1], 3))

    if augmentation:
        tf_image = tf.image.random_brightness(tf_image, max_delta=random.uniform(0, 1))
        tf_image = tf.image.random_saturation(tf_image, lower=0.5, upper=1.5)
        tf_image = tf.image.random_contrast(tf_image, lower=0.5, upper=1.5)
        tf_image = tf.image.convert_image_dtype(tf_image, dtype=tf.float32)

    tf_image_gray = tf.image.rgb_to_grayscale(tf_image)

    if apply_sobel_bool:
        return apply_sobel(tf_image_gray, ratio=ratio)
    else:
        return tf_image_gray


def rgb_to_grayscale(image):
    return tf.image.rgb_to_grayscale(image)


def apply_sobel(tf_image, ratio=0.4):
    tf_image = tf.image.sobel_edges(tf_image)
    tf_image = tf.squeeze(tf_image)
    return tf.where(tf_image > ratio, tf.ones_like(tf_image), tf.zeros_like(tf_image))


def flipping_tensor(tf_image):
    """
    flipping the image to double the size of the dataset and to balanced a bit more the dataset regarding turns
    """
    tf_image = tf.concat([tf_image, tf.image.flip_left_right(tf_image)], axis=0)
    return tf_image


def flipping_target(target):
    """
    If we choose to flip the image, we must also switch the targets with right --> left and left --> right
    """
    target = np.array(target)
    for i, t in enumerate(target):
        t = np.array(t)
        argmax = t.argmax()
        if argmax in [2, 4, 6]:
            target[i][argmax + 1] = 1
            target[i][argmax] = 0
        elif argmax in [3, 5, 7]:
            target[i][argmax - 1] = 1
            target[i][argmax] = 0
    return target


def process_image(image, resize):
    """
    process an image. There is no need of data augmentation
    """
    image = image / 255
    im_tensor = resize_and_scale_gray(image, resize, augmentation=False)
    im_tensor = tf.image.crop_to_bounding_box(
        im_tensor, offset_height=120, offset_width=0, target_height=150, target_width=480
    )

    im_tensor = tf.image.resize(im_tensor, resize)
    return tf.reshape(im_tensor[:, :, 0], (-1, resize[0], resize[1], 1))

def digit_images(tf_image,  digit_width = 20):


    tf_image_contrasted = tf.image.adjust_contrast(tf_image, 50)
    tf_image_contrasted = rgb_to_grayscale(tf_image_contrasted)
    first_digit = tf.image.crop_to_bounding_box(
        tf_image_contrasted, offset_height=0, offset_width=5, target_height=30, target_width=digit_width)
    second_digit = tf.image.crop_to_bounding_box(
        tf_image_contrasted, offset_height=0, offset_width=25, target_height=30, target_width=digit_width)
    third_digit = tf.image.crop_to_bounding_box(
        tf_image_contrasted, offset_height=0, offset_width=45, target_height=30, target_width=digit_width)

    return first_digit, second_digit, third_digit

def process(data_path, resize, augmentation=False, flipping_data=True):
    """
    Process the whole database contained at datapath
    """
    im_tensor = read_image(data_path)
    im_tensor = resize_and_scale_gray(im_tensor, resize, augmentation=augmentation)
    im_tensor = tf.image.crop_to_bounding_box(
        im_tensor, offset_height=120, offset_width=0, target_height=150, target_width=480
    )
    im_tensor = tf.image.resize(im_tensor, resize)

    if flipping_data:
        im_tensor = flipping_tensor(im_tensor)

    return tf.reshape(im_tensor[:, :, :, 0], (-1, resize[0], resize[1], 1))
