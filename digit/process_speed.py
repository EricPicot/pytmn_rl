import numpy as np
import tf2_processing

digit_width = 20
digit_db = []

#  Speed_data is a database of 50 screenshot of the 3 digits speed
speed_images_path = "speed_digit_data/speed_data.npy"
speed = np.load(speed_images_path)


#  Let's transform three digit number image into three images of one digit each.
# Appending it to database
for image in speed:
    first_digit, second_digit, third_digit = tf2_processing.digit_images(image)
    digit_db.append([first_digit])
    digit_db.append([second_digit])
    digit_db.append([third_digit])

np.save("speed_digit_data/digit_data.npy", digit_db)

