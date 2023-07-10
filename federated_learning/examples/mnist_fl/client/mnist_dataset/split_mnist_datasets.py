import os
import random
from shutil import copy


def split_mnist_datasets(dataset_path, clients):
    digit_folder_paths = os.listdir(dataset_path)
    dataset_dict = {}

    for digit in digit_folder_paths:
        digit_images_global_path = dataset_path + '/' + digit
        for image_name in os.listdir(digit_images_global_path):
            image_global_path = dataset_path + '/' + digit + '/' + image_name
            dataset_dict[image_global_path] = int(digit)

    l = list(dataset_dict.items())
    random.shuffle(l)
    dataset_dict = dict(l)

    samples_per_client = len(dataset_dict) // clients

    for client_i in range(clients):
        client_path = f'client_{client_i + 1}'
        os.mkdir(client_path)
        client_path += '/'
        for digit in range(10):
            os.mkdir(client_path + str(digit))

        image_paths = list(dataset_dict.keys())

        for sample_i in range(samples_per_client):
            item_i = client_i * samples_per_client + sample_i

            image_path = image_paths[item_i]
            image_name = image_path.split('/')[-1]
            digit = dataset_dict[image_path]

            copy(image_path, client_path + str(digit) + '/' + image_name)



split_mnist_datasets(dataset_path='trainingSet/trainingSet', clients=10)
