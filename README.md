# Pick-a-number 

#Preferiblemente correr este programa en linux

Como utilizar:
--------------
Luego de instalar requisitos:
	Desde el directorio Pick-a-number usar los siguientes comandos:
	
	python main.py

	Despues de que la red entrene se le pedira la ruta de la imagen a procesar
	esta debe ser de 10*10px se incluyen varias imagenes para probar en la carpeta
	prueba (estas imagenes son diferentes de las de entrenamiento) si la imagen esta en la carpeta desde donde se ejecuto main.py simplemente escribir
	 <nombre de imagen>.pbm en caso contrario poner direccion de la imagen completa.  

--------------------------------------------------------------
-Se utilizó una red Perceptron multicapa, utilizando Sigmoides como función de activación en la capa oculta.

-La imagen de entrada se representó con una matriz que luego fue convertida a arreglo, donde cada elemento de dicho arreglo es un parametro de entrada. Se llegó a esta decisión al pensar en evaluar todos los elementos de la matriz en la misma capa.

-Se utilizaron 10 neuronas de salida, cada una representando un número del 0 al 9. La neurona con el número mayor es el supuesto número a representar. Se llegó a esta conclusión luego de investigaciones que sugerían era el mejor método, ya que al utilizar una sola neurona se perdía parte de la representación y al usar 4 neuronas que representaban 2^4 los resultados podían ser más complejos de leer e inconclusos.

-Se dispusieron de 1/3 del número de neuronas de entrada como neuronas ocultas. Al principio se utilizó el mismo número de neuronas de entrada pero el porcentaje de probabilidad era muy bajo, y al colocar 1/10 el algoritmo se tardaba mucho en dar un estimado. Al colocar 1/3 obtenemos un porcentaje aceptable de éxitos sin mucho tiempo de procesamiento.


REQUISITOS: (Modulos de python)


python-opencv (Descargado desde el gestor de paquetes del sistema operativo)

pybrain 

scipy (dependencia de pybrain) 

numpy (dependencia de pybrain)
