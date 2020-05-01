from abc import ABC
import time
import tf2_processing
from utils import grab_screen, prediction_to_keys, speed_numerisation
import cv2
import tensorflow as tf
from gym import Env, spaces
import keyboard as kb
import os

print(os.listdir())
digit_model = tf.keras.models.load_model("./digit/models/digit_model")
starting_value = 1
speed_region = (730, 570, 800, 600)
screen_region = (0, 45, 800, 645)
HEIGHT = 270
WIDTH = 480
done_value = 10


class GymTMU(Env, ABC):
    def __init__(self):
        # observation_space
        # An image of size 270 * 480
        self.null_speed = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=[HEIGHT, WIDTH, 1])
        # 32 cards
        self.action_space = spaces.Box(low=0, high=1, shape=(9,))
        # self.action_space = spaces.discrete.Discrete(9)


        kb.press_and_release("del")


    def reset(self):
        """
        reset is mandatory to use gym framework. Reset is called at the end of each round (8 tricks)
        :return: observation
        """
        print("In reset")
        kb.press_and_release("del")
        self.null_speed = 0
        screen, speed = grab_screen(region=screen_region, speed_region=speed_region)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        print(screen.shape)
        print("avant process")
        observation = tf2_processing.process_image(screen, resize=(HEIGHT, WIDTH))
        print(observation.shape)
        observation = tf.reshape(observation[0, :, :, 0], (HEIGHT, WIDTH))
        print("fin de reset")
        return observation

    def step(self, action):
        """
        step is mandatory to use gym framework
        In each step, every player play exactly one, especially the AIPlayers.
        :param action:
        :return: observation, reward, done, info
        """
        key = prediction_to_keys(action)
        kb.press(key)
        screen, speed = grab_screen(region=screen_region, speed_region=speed_region)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        screen = tf2_processing.process_image(screen, resize=(HEIGHT, WIDTH))
        observation = tf.reshape(screen[0,:, :, 0], (HEIGHT, WIDTH))
        reward = speed_numerisation(speed, model=digit_model)
        kb.release(key)
        print(reward)
        done = False
        if reward < 20:
            print(self.null_speed)
            self.null_speed += 1
        else:
            self.null_speed = 0

        if self.null_speed >= done_value:
            reward = -10
            done = True
        info = {}
        return observation, reward, done, info