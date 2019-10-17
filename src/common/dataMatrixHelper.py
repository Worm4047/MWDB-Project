import numpy as np
import os

def read_data_matrix(model, folder):
    if not os.path.exists(get_path(model, folder)):
        return None
    else:
        np.genfromtxt(get_path(model, folder), delimiter=',')

def get_path(model, folder):
    return folder + "/dataMatrix/"  +'_'+ model + ".csv"

def save_data_matrix(model, folder, data_matrix):
    np.savetxt(get_path(model, folder), data_matrix, delimiter=",")

def filter_by_label(data_matrix, all_images, images_with_label):
    indexes = [all_images.index(i) for i in images_with_label]
    indexes.sort()
    return data_matrix[indexes, :]

