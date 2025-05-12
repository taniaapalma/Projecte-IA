__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset, read_extended_dataset, crop_images
from Kmeans_modificat import * 
import matplotlib as plt


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    options = {'fitting':'Inter'}
    km = KMeans(test_color_labels,1,options)
    km.fit()
    inter_values = km.Inter
    Ks = [2,len(inter_values)]
    plt.figure(figsize=(8, 5))
    plt.plot(Ks, inter_values, marker='o', color='royalblue', label='Interclass Distance')
    plt.xlabel('Número de Clústers (K)')
    plt.ylabel('Distancia Interclass')
    plt.title('Distancia Interclass vs Número de Clústers')
    plt.grid(True)
    plt.legend()
    plt.show()