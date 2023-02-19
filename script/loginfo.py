__author__ = "Keenan Manpearl"
__date__ = "2023/1/24"

"""
original code by Xinqiang Ding <xqding@umich.edu>

records loss
"""

import matplotlib.pyplot as plt

loss = []
with open("./log/loginfo_new.txt", "r") as file_handle:
    for line in file_handle:
        line = line.strip()
        field = line.split()
        if field[-1] != "nan":
            loss.append(float(field[-1]))

plt.plot(loss[::30])
plt.ylim(70, 200)
plt.show()
