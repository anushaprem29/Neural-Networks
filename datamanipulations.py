import csv
import numpy as np


def create_csv():
    input_path = "data.csv"
    vc_path = "Vc.csv"
    vh_path = "Vh.csv"
    with open(input_path) as fp, open(vc_path, 'wb') as vc_path, open(vh_path, 'wb') as vh_path:
        input_reader = csv.reader(fp, delimiter=',')
        vc_write = csv.writer(vc_path)
        vh_write = csv.writer(vh_path)
        for row in input_reader:
            vc_write.writerow(row[1:11])
            vh_write.writerow(row[11:])