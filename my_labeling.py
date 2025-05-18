__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset, read_extended_dataset, crop_images
from Kmeans_modificat import * 
from KNN import * 
import matplotlib.pyplot as plt
import math


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='images/', gt_json='images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here

    """
    # FUNCIONES RETRIEVAL

     VERSION SILVIA:
    def Retrieval_by_color(image_list, image_color_labels, query, image_color_percentages=None):
    # Si query es una cadena, convertirla a una lista con un solo elemento

        if isinstance(query, str):
            query = [query]

        # Convertir todos los colores de consulta a minúsculas
        query_lower = [str(color).lower() for color in query]

        # Lista imágenes que coincidan con los colores
        matching_indices = []

        for i, label in enumerate(image_color_labels):
            matching_colors = [color for color in label if color in query] # Encontrar los colores en la imagen que también están en los colores de la consulta
            if matching_colors:
                relevance_score = 0
                if image_color_percentages is not None:
                    relevance_score = sum(image_color_percentages[i].get(color, 0) for color in matching_colors)

                matching_indices.append((relevance_score, image_list[i]))

        if image_color_percentages is not None:
            matching_indices.sort(key=lambda x: x[0], reverse=True) # ordenar las imágenes por el score en orden descendente

        #return [img for _, img in matching_indices] if matching_indices and isinstance(matching_indices[0], tuple) else matching_indices

        visualitzar_retrieval(image_list, matching_indices, "Busqueda por el color ", query)

    """

    def Retrieval_by_color(llista_imatges, etiquetes, query, image_color_percentages=None):
        if isinstance(query, str):
            query = [query]
        cerca = [str(color).lower() for color in query]
        matching_indices = []
        for i, label in enumerate(etiquetes):
            if isinstance(label, str):
                label = [label]
            label_lower = [str(c).lower() for c in label]
            matching_colors = [color for color in label_lower if color in cerca]
            if matching_colors:
                relevance_score = 0
                if image_color_percentages is not None:
                    relevance_score = sum(image_color_percentages[i].get(color, 0) for color in matching_colors)
                matching_indices.append((relevance_score, i))  # Guardar índice, no imagen
        if image_color_percentages is not None:
            matching_indices.sort(key=lambda x: x[0], reverse=False)
        result_indices = [idx for _, idx in matching_indices] if matching_indices else []

        # Mostrar resultados
        visualitzar_retrieval(llista_imatges, result_indices, "Búsqueda por color", query)









    def Retrieval_by_shape(llista_imatges, etiquetes, query):

        if isinstance(query, str):
            query = [query]
        cerca = [str(color).lower() for color in query]
        matching_indices = []

        for i, label in enumerate(etiquetes):
            if str(label).lower() == cerca:
                matching_indices.append(i)

        if not matching_indices:
            print(f"No images match the query '{cerca}'")
        else:
            visualitzar_retrieval(llista_imatges, matching_indices, "Busqueda por forma: ", query)




    def Retrieval_combined(llista_imatges, shape_labels, color_labels, query_shape, query_color):
        if isinstance(query_shape, str):
            query_shape = [query_shape]
        if isinstance(query_color, str):
            query_color = [query_color]

        # Pasar todo a minúsculas
        query_shape = [str(s).lower() for s in query_shape]
        query_color = [str(c).lower() for c in query_color]

        shape_matches = set()
        color_matches = set()

        # Encontrar índices de forma
        for i, label in enumerate(shape_labels):
            if str(label).lower() in query_shape:
                shape_matches.add(i)

        # Encontrar índices de color
        for i, label in enumerate(color_labels):
            if isinstance(label, str):
                label = [label]  # Convertir en llista si cal
            label = [str(c).lower() for c in label]
            if any(color in query_color for color in label):
                color_matches.add(i)

        # Cogemos solo las que tengan ambas
        combined_matches = sorted(list(shape_matches & color_matches))

        if not combined_matches:
            print(f"No se encontraron imágenes que coincidan con: forma(s) {query_shape} y color(es) {query_color}")

        # Mostrar resultados
        visualitzar_retrieval(llista_imatges, combined_matches, "Búsqueda combinada", f"Forma: {query_shape}, Color: {query_color}")






    # FUNCION PARA VISUALIZAR -------------------------------------------------------------
    def visualitzar_retrieval(imgs, indices, nom, query_info=None):
        if not indices:
            print(f"Ninguna imagen coincide con: {nom}")
            return
    
        # Configurar la visualización
        n = len(indices)
        cols = min(n, 4)  # Máximo 4 columnas
        rows = math.ceil(n / cols)
        
        # Crear la figura y los ejes
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows * cols == 1:
            axes = np.array([axes])  # Convertir a array para manejar el caso de una sola imagen
        else:
            axes = np.array(axes).reshape(-1)  # Asegurar que axes siempre sea 1D
        
        # Mostrar cada imagen coincidente
        for idx, img_idx in enumerate(indices):
            axes[idx].imshow(imgs[img_idx])
            axes[idx].set_title(f"ID: {img_idx}", fontsize=10)
            axes[idx].axis('off')
        
        # Ocultar los subplots vacíos si hay alguno
        for j in range(len(indices), len(axes)):
            axes[j].axis('off')
        
        # Configurar el título y ajustar el diseño
        full_title = nom
        if query_info:
            full_title += f": {query_info}"
        
        plt.suptitle(full_title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Dejar espacio para el título
        plt.show()

    
    
    
    
    

    def Get_shape_accuracy(predicted_labels, ground_truth_labels):
        
        # Calcular la precisión para cada imagen (1 si coincide, 0 si no)
        if len(predicted_labels) != len(ground_truth_labels):
            raise ValueError("Las listas de etiquetas deben tener la misma longitud.")

        correctas = 0
        total = len(predicted_labels)

        for pred, real in zip(predicted_labels, ground_truth_labels):
            if pred == real:
                correctas += 1

        return correctas / total if total > 0 else 0.0
    

    #Pruebas--------------------------------------------------
            
    #Retrieval_by_color(imgs, color_labels, 'red')
    #Retrieval_by_shape(imgs, class_labels, "heels")
    #Retrieval_combined(imgs, class_labels, color_labels, "shirts", "red")

    """
    knn = KNN(train_imgs, train_class_labels)
    predicted = knn.predict(test_imgs, k=1)

    accuracy = Get_shape_accuracy(predicted, test_class_labels)
    print(f"Precisión de forma: {accuracy * 100}%")

    """

    k_values = list(range(1, 13))  # De k=1 a k=12
    accuracies = []

    # Ejecutar pruebas
    for k in k_values:
        print(f"Probando con K={k}...")
        knn = KNN(train_imgs, train_class_labels)
        predicted = knn.predict(test_imgs, k=k)
        accuracy = Get_shape_accuracy(predicted, test_class_labels)
        accuracies.append(accuracy)
        print(f"Precisión para k={k}: {accuracy * 100}%")

    # Mostrar gráfica
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, [a * 100 for a in accuracies], marker='o', color='blue')
    plt.title('Precisión en función de K (clasificación de forma)')
    plt.xlabel('K (número de vecinos)')
    plt.ylabel('Precisión (%)')
    plt.grid(True)
    plt.xticks(k_values)
    plt.ylim(0, 100)
    plt.show()
