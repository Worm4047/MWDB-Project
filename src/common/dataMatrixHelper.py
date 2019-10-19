import numpy as np
import os

def read_data_matrix(model, label, folder):
    if not os.path.exists(get_path(model, label, folder)):
        print("Computing Data Matrix")
        return None
    else:
        print("Retrived Data Matrix")
        return np.genfromtxt(get_path(model, label, folder), delimiter=',')

def get_path(model, label, folder):
    return os.path.join(folder, "{}_{}.csv".format(model.name, label))

def save_data_matrix(model, label, folder, data_matrix):
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.savetxt(get_path(model, label, folder), data_matrix, delimiter=",")

def filter_by_label(data_matrix, all_images, images_with_label):
    indexes = []
    for i in images_with_label:
        if i in all_images:
            indexes.append(all_images.index(i))
    indexes.sort()
    return data_matrix[indexes, :]

