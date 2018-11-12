import csv
import numpy as np

split_point = 2010


def create_csv():
    input_data = []
    output_data = []
    path = "data/data.csv"
    with open(path) as fp:
        input_reader = csv.reader(fp, delimiter=',')
        for row in input_reader:
            if len(row) > 0:
                input_data.append(row[1:11])
                output_data.append(row[11:])
    return np.array(input_data[1:]), np.array(output_data[1:])


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