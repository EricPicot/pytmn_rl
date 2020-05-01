import numpy as np
import matplotlib.pyplot as plt

target = []
plt.ion()
digit_data = np.load("speed_digit_data/digit_data.npy")
print("Tagging manually {} digit image".format(len(digit_data)))
for digit_image in digit_data:
    plt.imshow(digit_image[:, :, 0])
    plt.show(block=False)
    input("press enter")
    plt.close()
    x = input("value")
    print(x)
    target.append(x)
    print(target)

np.save("./tf_dataset/speed_digit_data/digit_target2.npy", target)
