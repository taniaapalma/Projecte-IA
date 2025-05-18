import numpy as np
import matplotlib.pyplot as plt
import time
from Kmeans_modificat import KMeans, get_colors

def accuracy_combinada_simple(imagenes, colores_reales, max_K, espacio_11d=False):
    """
    Evalúa la accuracy para todas las combinaciones de métodos de inicialización
    y heurísticas de KMeans, mostrando resultados en tablas simples.
    
    Args:
        imagenes: Lista de imágenes a procesar
        colores_reales: Lista de colores reales para cada imagen
        max_K: Número máximo de clusters a probar
        espacio_11d: Si True, usa el espacio de 11 dimensiones
        
    Returns:
        dict: Resultados de accuracy para todas las combinaciones
    """
    # Definir métodos de inicialización y heurísticas
    metodos_init = ['first', 'random', 'farthest', 'uniform']
    heuristicas = ['wcd', 'inter', 'fisher']
    
    # Inicializar matrices de resultados
    accuracy_colores = np.zeros((len(metodos_init), len(heuristicas)))
    accuracy_tamaño = np.zeros((len(metodos_init), len(heuristicas)))
    tiempo_ejecucion = np.zeros((len(metodos_init), len(heuristicas)))
    k_optimo = np.zeros((len(metodos_init), len(heuristicas)))
    
    # Para cada combinación de método y heurística
    for i, metodo in enumerate(metodos_init):
        for j, heuristica in enumerate(heuristicas):
            print(f"Evaluando combinación: {metodo} + {heuristica}")
            
            # Configurar opciones
            opciones = {
                'km_init': metodo,
                'fitting': heuristica,
                '11': espacio_11d
            }
            
            # Medir tiempo de inicio
            tiempo_inicio = time.time()
            
            # Contadores para accuracy
            contadorColores = 0
            contadorTamaño = 0
            k_optimos = []
            
            # Procesar cada imagen
            for idx in range(len(imagenes)):
                try:
                    # Crear y entrenar KMeans
                    km = KMeans(imagenes[idx], max_K, opciones)
                    km.find_bestK(max_K)
                    coloresKM = get_colors(km.centroids, espacio_11d)
                    
                    # Guardar K óptimo
                    k_optimos.append(km.K)
                    
                    # Verificar tamaño
                    if len(coloresKM) == len(colores_reales[idx]):
                        contadorTamaño += 1
                        
                        # Comparar frecuencias de colores
                        colores_pred_freq = {}
                        colores_real_freq = {}
                        
                        for color in coloresKM:
                            colores_pred_freq[color] = colores_pred_freq.get(color, 0) + 1
                        
                        for color in colores_reales[idx]:
                            colores_real_freq[color] = colores_real_freq.get(color, 0) + 1
                        
                        # Verificar si las frecuencias coinciden
                        if colores_pred_freq == colores_real_freq:
                            contadorColores += 1
                except Exception as e:
                    print(f"Error en imagen {idx} con {metodo}+{heuristica}: {str(e)}")
                    continue
            
            # Calcular accuracies
            accuracy_colores[i, j] = 100 * contadorColores / len(imagenes)
            accuracy_tamaño[i, j] = 100 * contadorTamaño / len(imagenes)
            
            # Medir tiempo total
            tiempo_ejecucion[i, j] = time.time() - tiempo_inicio
            
            # Calcular K óptimo promedio
            k_optimo[i, j] = np.mean(k_optimos) if k_optimos else 0
            
            print(f"  Accuracy colores: {accuracy_colores[i, j]:.2f}%")
            print(f"  Accuracy tamaño: {accuracy_tamaño[i, j]:.2f}%")
            print(f"  Tiempo: {tiempo_ejecucion[i, j]:.2f}s")
            print(f"  K óptimo promedio: {k_optimo[i, j]:.2f}")
    
    # Mostrar resultados en tablas
    mostrar_resultados_en_tablas(
        accuracy_colores, accuracy_tamaño, tiempo_ejecucion, k_optimo,
        metodos_init, heuristicas
    )
    
    # Mostrar gráfico de barras para las mejores combinaciones
    mostrar_mejores_combinaciones(
        accuracy_colores, accuracy_tamaño, tiempo_ejecucion, k_optimo,
        metodos_init, heuristicas
    )
    
    # Devolver resultados como diccionario para uso posterior
    resultados = {
        'accuracy_colores': accuracy_colores,
        'accuracy_tamaño': accuracy_tamaño,
        'tiempo_ejecucion': tiempo_ejecucion,
        'k_optimo': k_optimo,
        'metodos_init': metodos_init,
        'heuristicas': heuristicas
    }
    
    return resultados

def mostrar_resultados_en_tablas(accuracy_colores, accuracy_tamaño, tiempo_ejecucion, k_optimo, metodos_init, heuristicas):
    """
    Muestra los resultados en tablas usando matplotlib.
    
    Args:
        accuracy_colores: Matriz con accuracy de colores
        accuracy_tamaño: Matriz con accuracy de tamaño
        tiempo_ejecucion: Matriz con tiempos de ejecución
        k_optimo: Matriz con K óptimo promedio
        metodos_init: Lista de métodos de inicialización
        heuristicas: Lista de heurísticas
    """
    # Crear figura para las tablas
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Resultados de Accuracy Combinada', fontsize=16)
    
    # Tabla 1: Accuracy de Colores
    ax1 = axs[0, 0]
    ax1.axis('tight')
    ax1.axis('off')
    tabla1 = ax1.table(
        cellText=[[f"{val:.1f}%" for val in row] for row in accuracy_colores],
        rowLabels=metodos_init,
        colLabels=heuristicas,
        loc='center',
        cellLoc='center'
    )
    tabla1.auto_set_font_size(False)
    tabla1.set_fontsize(10)
    tabla1.scale(1.2, 1.5)
    ax1.set_title('Accuracy de Colores (%)')
    
    # Tabla 2: Accuracy de Tamaño
    ax2 = axs[0, 1]
    ax2.axis('tight')
    ax2.axis('off')
    tabla2 = ax2.table(
        cellText=[[f"{val:.1f}%" for val in row] for row in accuracy_tamaño],
        rowLabels=metodos_init,
        colLabels=heuristicas,
        loc='center',
        cellLoc='center'
    )
    tabla2.auto_set_font_size(False)
    tabla2.set_fontsize(10)
    tabla2.scale(1.2, 1.5)
    ax2.set_title('Accuracy de Tamaño (%)')
    
    # Tabla 3: Tiempo de Ejecución
    ax3 = axs[1, 0]
    ax3.axis('tight')
    ax3.axis('off')
    tabla3 = ax3.table(
        cellText=[[f"{val:.2f}s" for val in row] for row in tiempo_ejecucion],
        rowLabels=metodos_init,
        colLabels=heuristicas,
        loc='center',
        cellLoc='center'
    )
    tabla3.auto_set_font_size(False)
    tabla3.set_fontsize(10)
    tabla3.scale(1.2, 1.5)
    ax3.set_title('Tiempo de Ejecución (segundos)')
    
    # Tabla 4: K Óptimo Promedio
    ax4 = axs[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    tabla4 = ax4.table(
        cellText=[[f"{val:.1f}" for val in row] for row in k_optimo],
        rowLabels=metodos_init,
        colLabels=heuristicas,
        loc='center',
        cellLoc='center'
    )
    tabla4.auto_set_font_size(False)
    tabla4.set_fontsize(10)
    tabla4.scale(1.2, 1.5)
    ax4.set_title('K Óptimo Promedio')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def mostrar_mejores_combinaciones(accuracy_colores, accuracy_tamaño, tiempo_ejecucion, k_optimo, metodos_init, heuristicas):
    """
    Muestra un gráfico de barras con las mejores combinaciones.
    
    Args:
        accuracy_colores: Matriz con accuracy de colores
        accuracy_tamaño: Matriz con accuracy de tamaño
        tiempo_ejecucion: Matriz con tiempos de ejecución
        k_optimo: Matriz con K óptimo promedio
        metodos_init: Lista de métodos de inicialización
        heuristicas: Lista de heurísticas
    """
    # Crear lista de todas las combinaciones
    combinaciones = []
    valores_colores = []
    valores_tamaño = []
    
    for i, metodo in enumerate(metodos_init):
        for j, heuristica in enumerate(heuristicas):
            combinaciones.append(f"{metodo}\n{heuristica}")
            valores_colores.append(accuracy_colores[i, j])
            valores_tamaño.append(accuracy_tamaño[i, j])
    
    # Ordenar por accuracy de colores
    indices_ordenados = np.argsort(valores_colores)[::-1]  # Orden descendente
    combinaciones_ordenadas = [combinaciones[i] for i in indices_ordenados]
    valores_colores_ordenados = [valores_colores[i] for i in indices_ordenados]
    valores_tamaño_ordenados = [valores_tamaño[i] for i in indices_ordenados]
    
    # Crear gráfico de barras
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(combinaciones_ordenadas))
    width = 0.35
    
    plt.bar(x - width/2, valores_colores_ordenados, width, label='Accuracy Colores')
    plt.bar(x + width/2, valores_tamaño_ordenados, width, label='Accuracy Tamaño')
    
    plt.xlabel('Combinación (método+heurística)')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparación de Accuracy por Combinación')
    plt.xticks(x, combinaciones_ordenadas, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Añadir valores sobre las barras
    for i, v in enumerate(valores_colores_ordenados):
        plt.text(i - width/2, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=8)
    
    for i, v in enumerate(valores_tamaño_ordenados):
        plt.text(i + width/2, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar tabla con las 5 mejores combinaciones
    mostrar_top_combinaciones(
        accuracy_colores, accuracy_tamaño, tiempo_ejecucion, k_optimo,
        metodos_init, heuristicas, top_n=5
    )

def mostrar_top_combinaciones(accuracy_colores, accuracy_tamaño, tiempo_ejecucion, k_optimo, metodos_init, heuristicas, top_n=5):
    """
    Muestra una tabla con las mejores combinaciones.
    
    Args:
        accuracy_colores: Matriz con accuracy de colores
        accuracy_tamaño: Matriz con accuracy de tamaño
        tiempo_ejecucion: Matriz con tiempos de ejecución
        k_optimo: Matriz con K óptimo promedio
        metodos_init: Lista de métodos de inicialización
        heuristicas: Lista de heurísticas
        top_n: Número de mejores combinaciones a mostrar
    """
    # Crear lista de todas las combinaciones con sus métricas
    datos = []
    
    for i, metodo in enumerate(metodos_init):
        for j, heuristica in enumerate(heuristicas):
            datos.append({
                'metodo': metodo,
                'heuristica': heuristica,
                'accuracy_colores': accuracy_colores[i, j],
                'accuracy_tamaño': accuracy_tamaño[i, j],
                'tiempo': tiempo_ejecucion[i, j],
                'k_optimo': k_optimo[i, j]
            })
    
    # Ordenar por accuracy de colores
    datos_ordenados = sorted(datos, key=lambda x: x['accuracy_colores'], reverse=True)
    
    # Seleccionar las top_n mejores combinaciones
    mejores = datos_ordenados[:top_n]
    
    # Preparar datos para la tabla
    filas = []
    for dato in mejores:
        filas.append([
            dato['metodo'],
            dato['heuristica'],
            f"{dato['accuracy_colores']:.1f}%",
            f"{dato['accuracy_tamaño']:.1f}%",
            f"{dato['tiempo']:.2f}s",
            f"{dato['k_optimo']:.1f}"
        ])
    
    # Crear figura para la tabla
    plt.figure(figsize=(12, 4))
    plt.axis('off')
    
    # Crear tabla
    tabla = plt.table(
        cellText=filas,
        colLabels=['Método', 'Heurística', 'Accuracy Colores', 'Accuracy Tamaño', 'Tiempo (s)', 'K Óptimo'],
        loc='center',
        cellLoc='center'
    )
    
    # Ajustar tamaño de la tabla
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 1.5)
    
    plt.title(f'Top {top_n} Mejores Combinaciones', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    from utils_data import *
    
    # Cargar datos
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='images/', gt_json='images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)
    
    # Ejecutar con un subconjunto pequeño para pruebas
    resultados = accuracy_combinada_simple(test_imgs, test_color_labels, max_K=10, espacio_11d=True)


    # Acceder a los resultados
    mejor_combinacion = max(resultados['accuracy_colores'], key=resultados['accuracy_colores'].get)
    print(f"Mejor combinación: {mejor_combinacion}")
    print(f"Accuracy de colores: {resultados['accuracy_colores'][mejor_combinacion]:.2f}%")
    print(f"Accuracy de tamaño: {resultados['accuracy_tamaño'][mejor_combinacion]:.2f}%")
    print(f"Tiempo de ejecución: {resultados['tiempo_ejecucion'][mejor_combinacion]:.2f}s")
    print(f"K óptimo promedio: {resultados['K_optimo'][mejor_combinacion]:.2f}")


    # You can start coding your functions here
    def graficaInter(train_img, max_K, grafica = 0):
      options = {
      'fitting': 'Inter',
      'km_init': 'random'
      }

      km = KMeans(train_img, max_K, options)
      km.find_bestK(max_K)

      if grafica:
        inter_values = km.Inter
        Ks = list(range(2, 2 + len(inter_values)))  # <-- Este debe coincidir con los valores reales de Inter

        plt.figure(figsize=(8, 5))
        plt.plot(Ks, inter_values, marker='o', color='royalblue', label='Distancia Interclass')
        plt.axvline(km.K, color='green', linestyle='--', label=f'K óptimo = {km.K}')
        plt.xlabel('Número de Clústers (K)')
        plt.ylabel('Distancia Interclass')
        plt.title('Distancia Interclass vs Número de Clústers')
        plt.grid(True)
        plt.legend()
        plt.show()
      
      return km.centroids
    
    def graficaWCD(train_img, max_K, grafica=0):
      options = {
      'fitting': 'WCD',
      'km_init': 'random'
      }

      km = KMeans(train_img, max_K, options)
      km.find_bestK(max_K)

      if grafica:
        inter_values = km.WCD
        Ks = list(range(2, 2 + len(inter_values)))  # <-- Este debe coincidir con los valores reales de Inter

        plt.figure(figsize=(8, 5))
        plt.plot(Ks, inter_values, marker='o', color='royalblue', label='Distancia Intraclass')
        plt.axvline(km.K, color='green', linestyle='--', label=f'K óptimo = {km.K}')
        plt.xlabel('Número de Clústers (K)')
        plt.ylabel('Distancia Intraclass')
        plt.title('Distancia Intraclass vs Número de Clústers')
        plt.grid(True)
        plt.legend()
        plt.show()

      return km.centroids


    def graficaFisher(train_img, max_K, grafica=0):
      options = {
      'fitting': 'fisher',
      'km_init': 'random'
      }
      km = KMeans(train_img, max_K, options)
      km.find_bestK(max_K)
      km.fit()
      if grafica:
        fisher_values = km.fish
        Ks = list(range(2, 2 + len(fisher_values)))  # <-- Este debe coincidir con los valores reales de Inter
        plt.figure(figsize=(8, 5))
        plt.plot(Ks, fisher_values, marker='o', color='royalblue', label='Discriminante Fisher')
        plt.axvline(km.K, color='green', linestyle='--', label=f'K óptimo = {km.K}')
        plt.xlabel('Número de Clústers (K)')
        plt.ylabel('Discriminante Fisher')
        plt.title('Discriminante Fisher vs Número de Clústers')
        plt.grid(True)
        plt.legend()
        plt.show()
      
      return km.centroids
    
    #graficaFisher(train_imgs[0],1)
    #graficaInter(train_imgs[0],1)
    #graficaWCD(train_imgs[0],1)

    def comparaHeu(train_imgs, colores, max_K):
      contadorInter = 0
      contadorWCD = 0
      contadorFisher = 0
      for i in range(len(train_imgs)):
        centroidesInter = graficaInter(train_imgs[i], max_K, 0)
        centroidesIntra = graficaWCD(train_imgs[i], max_K, 0)
        centroidesFisher = graficaFisher(train_imgs[i], max_K, 0)

        if len(centroidesInter) == len(colores[i]):
          contadorInter += 1
        if len(centroidesIntra) == len(colores[i]):
          contadorWCD += 1
        if len(centroidesFisher) == len(colores[i]):
          contadorFisher += 1

      accuracyInter = contadorInter/len(train_imgs)
      accuracyWCD = contadorWCD/len(train_imgs)
      accuracyFisher = contadorFisher/len(train_imgs)

      heuristicas = ['Interclass', 'Intraclass (WCD)', 'Fisher']
      accuracies = [accuracyInter, accuracyWCD, accuracyFisher]

      # Crear figura y tabla
      fig, ax = plt.subplots()
      ax.axis('off')
      tabla = ax.table(cellText=[[f"{a:.2f}"] for a in accuracies],
                      rowLabels=heuristicas,
                      colLabels=["Accuracy"],
                      loc='center',
                      cellLoc='center')
      tabla.scale(1, 2)
      tabla.auto_set_font_size(False)
      tabla.set_fontsize(12)
      ax.set_title("Precisión de cada heurística", fontsize=14, pad=20)

      plt.show()
    

    def accuracy11Dim(testImg, colores, max_K, opciones):
      contadorColores = 0
      contadorTamaño = 0
      for i in range(len(testImg)):
        km = KMeans(testImg[i], max_K, opciones)
        km.find_bestK(max_K)
        coloresKM = get_colors(km.centroids, True)
        if len(coloresKM) == len(colores[i]):
          contadorTamaño += 1
          if set(coloresKM) == set(colores[i]):
            contadorColores += 1
      
      accuracy = 100*contadorColores/len(testImg)
      accuracy2 = 100*contadorTamaño/len(testImg)
      print(accuracy)
      print(accuracy2)

    def comparaKNNAccuracy(imagenes, max_K, testImagenes, etiquetas, tipos):

      accuracys = {}
      knn = KNN(imagenes, etiquetas)
      for i in range(1, max_K):
        correctos = 0
        for j in range(len(testImagenes)):
          clase = knn.predict(testImagenes[j], i)
          if clase == tipos[j]:
            correctos += 1
        accuracy = correctos/len(testImagenes)
        accuracys[i] = accuracy

      
      Ks = list(accuracys.keys())
      accuracys = list(accuracys.values())

      plt.figure(figsize=(8, 5))
      plt.plot(Ks, accuracys, marker='o', linestyle='-', color='blue')
      plt.xlabel('Valor de K')
      plt.ylabel('Accuracy')
      plt.title('Accuracy vs Valor de K (KNN)')
      plt.grid(True)
      plt.xticks(Ks)
      plt.ylim(0, 1.05)
      plt.show()

    def accuracy_colores(testImg, colores_reales, max_K, opciones):
      contadorColores = 0
      contadorTamaño = 0
      
      for i in range(len(testImg)):
          km = KMeans(testImg[i], max_K, opciones)
          km.find_bestK(max_K)
          coloresKM = get_colors(km.centroids, opciones.get('11', False))
          
          # Verificar si el número de colores coincide
          if len(coloresKM) == len(colores_reales[i]):
              contadorTamaño += 1
              
              # Comparar frecuencias de colores en lugar de solo conjuntos
              colores_pred_freq = {}
              colores_real_freq = {}
              
              for color in coloresKM:
                  colores_pred_freq[color] = colores_pred_freq.get(color, 0) + 1
              
              for color in colores_reales[i]:
                  colores_real_freq[color] = colores_real_freq.get(color, 0) + 1
              
              # Verificar si las frecuencias coinciden
              if colores_pred_freq == colores_real_freq:
                  contadorColores += 1
      
      accuracy = 100 * contadorColores / len(testImg)
      accuracy_tamaño = 100 * contadorTamaño / len(testImg)
      
      return accuracy, accuracy_tamaño
    
    def accuracyRGB(testImg, colores, max_K, opciones):
      contadorColores = 0
      contadorTamaño = 0
      for i in range(len(testImg)):
        km = KMeans(testImg[i], max_K, opciones)
        km.find_bestK(max_K)
        coloresKM = get_colors(km.centroids, False)
        if len(coloresKM) == len(colores[i]):
          contadorTamaño += 1
          if set(coloresKM) == set(colores[i]):
            contadorColores += 1
      
      accuracy = 100*contadorColores/len(testImg)
      accuracy2 = 100*contadorTamaño/len(testImg)
      print(accuracy)
      print(accuracy2)

    def Kmean_statistics(KMeansClass, images, Kmax, options):
      """
      Genera estadísticas de ejecución para KMeans con diferentes valores de K.
      
      Args:
          KMeansClass: Clase KMeans a utilizar
          images: Lista de imágenes a procesar
          Kmax: Número máximo de clusters a probar
          options: Opciones para KMeans
          
      Returns:
          dict: Diccionario con estadísticas (tiempos, iteraciones, WCD)
      """
      tiempos = []
      iteraciones = []
      wcds = []
      Ks = list(range(2, Kmax + 1))

      for k in Ks:
          total_time = 0
          total_iter = 0
          total_wcd = 0

          for img in images:
              km = KMeansClass(img, K=k, options=options)
              start = time.time()
              km.fit()
              end = time.time()

              total_time += (end - start)
              total_iter += km.num_iter
              total_wcd += km.withinClassDistance()

          # Promedios
          tiempos.append(total_time / len(images))
          iteraciones.append(total_iter / len(images))
          wcds.append(total_wcd / len(images))

      # Visualización de resultados
      plt.figure(figsize=(10, 6))
      plt.plot(Ks, tiempos, marker='o', color='royalblue')
      plt.title('Tiempo medio de ejecución del KMeans')
      plt.xlabel('K')
      plt.ylabel('Tiempo (segundos)')
      plt.grid(True)
      plt.show()  # Esta línea hace que se muestre la primera gráfica

      plt.figure(figsize=(10, 6))
      plt.plot(Ks, iteraciones, marker='s', color='green')
      plt.title('Iteraciones medias hasta convergir')
      plt.xlabel('K')
      plt.ylabel('Número de iteraciones')
      plt.grid(True)
      plt.show()  # Esta línea hace que se muestre la segunda gráfica

      plt.figure(figsize=(10, 6))
      plt.plot(Ks, wcds, marker='^', color='darkred')
      plt.title('Within Class Distance (WCD) medio')
      plt.xlabel('K')
      plt.ylabel('WCD')
      plt.grid(True)
      plt.show()
      
      return {
          'Ks': Ks,
          'tiempos': tiempos,
          'iteraciones': iteraciones,
          'wcds': wcds
      }
    

    def color_accuracy(testImg, colores_reales, max_K, opciones):
      """
      Calcula la precisión en la detección de colores.
      
      Args:
          testImg: Lista de imágenes a procesar
          colores_reales: Lista de colores reales para cada imagen
          max_K: Número máximo de clusters a probar
          opciones: Opciones para KMeans
          
      Returns:
          tuple: (accuracy_colores, accuracy_tamaño, detalles)
      """
      contadorColores = 0
      contadorTamaño = 0
      detalles = []
      
      for i in range(len(testImg)):
          km = KMeans(testImg[i], max_K, opciones)
          km.find_bestK(max_K)
          coloresKM = get_colors(km.centroids, opciones.get('11', False))
          
          # Verificar si el número de colores coincide
          mismo_tamaño = len(coloresKM) == len(colores_reales[i])
          if mismo_tamaño:
              contadorTamaño += 1
              
              # Comparar frecuencias de colores en lugar de solo conjuntos
              colores_pred_freq = {}
              colores_real_freq = {}
              
              for color in coloresKM:
                  colores_pred_freq[color] = colores_pred_freq.get(color, 0) + 1
              
              for color in colores_reales[i]:
                  colores_real_freq[color] = colores_real_freq.get(color, 0) + 1
              
              # Verificar si las frecuencias coinciden
              mismos_colores = colores_pred_freq == colores_real_freq
              if mismos_colores:
                  contadorColores += 1
          
          # Guardar detalles
          detalles.append({
              'imagen_idx': i,
              'K_encontrado': len(coloresKM),
              'K_real': len(colores_reales[i]),
              'colores_encontrados': coloresKM,
              'colores_reales': colores_reales[i],
              'mismo_tamaño': mismo_tamaño,
              'mismos_colores': mismo_tamaño and (colores_pred_freq == colores_real_freq if mismo_tamaño else False)
          })
      
      accuracy_colores = 100 * contadorColores / len(testImg)
      accuracy_tamaño = 100 * contadorTamaño / len(testImg)
      
      # Visualización de resultados
      fig, ax = plt.subplots(figsize=(8, 6))
      
      metricas = ['Accuracy Colores', 'Accuracy Tamaño']
      valores = [accuracy_colores, accuracy_tamaño]
      
      ax.bar(metricas, valores, color=['royalblue', 'green'])
      ax.set_ylim(0, 100)
      ax.set_ylabel('Accuracy (%)')
      ax.set_title('Precisión en la detección de colores')
      
      # Añadir valores sobre las barras
      for i, v in enumerate(valores):
          ax.text(i, v + 2, f"{v:.2f}%", ha='center')
      
      plt.tight_layout()
      plt.show()
      
      print(f"Accuracy de colores: {accuracy_colores:.2f}%")
      print(f"Accuracy de tamaño: {accuracy_tamaño:.2f}%")
      
      return accuracy_colores, accuracy_tamaño, detalles
      
    
    opciones = {
      'km_init': 'first',
      'fitting': 'fisher',
      '11': True
    }
    ##color_accuracy(train_imgs, train_color_labels, 10, opciones)
    ##Kmean_statistics(KMeans, imgs, 10, opciones)