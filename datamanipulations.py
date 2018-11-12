import csv
import numpy as np

split_point = 2010  # 80% of length.


def create_csv():
    input_data = []
    output_data = []

    path = "data/data.csv"
    with open(path) as fp:
        input_reader = list(csv.reader(fp, delimiter=','))
        for row in input_reader[1:]:
            if len(row) > 0:
                input_data.append(row[1:11])
                output_data.append(row[11:])
    in_array = np.array([[(float(s) if s else 0) for s in x] for x in input_data])
    out_array = np.array([[(float(s) if s else 0) for s in x] for x in output_data])

    return in_array, out_array


def load_data():
    input_data, output_data = create_csv()

    # shuffle data
    indices = np.arange(len(input_data))

    np.random.shuffle(indices)
    input_data = input_data[indices]
    output_data = output_data[indices]

    input_data_train = np.array(input_data[:split_point])
    output_data_train = np.array(output_data[:split_point])
    input_data_test = np.array(input_data[split_point:])
    output_data_test = np.array(output_data[split_point:])

    return (input_data_train, output_data_train), (input_data_test, output_data_test)
