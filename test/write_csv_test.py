import csv
import numpy as np

data = [[np.pi, 2, 3, 4], [np.e, 'b', 'c', 'd'], ['a', 'b', 'c', 'd']]

with open('test.csv', 'w+', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(data)
