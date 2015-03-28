#!/usr/bin/env python
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets.supervised import SupervisedDataSet

import os

import cv2

def load_data(size):
    ds = SupervisedDataSet(size, 1)
    #Agregando todos los elementos 0
    t = os.listdir(os.path.join('data','0'))
    for file in t:
        temp = os.path.join('data','0',file)
        ds.addSample(load_image(temp),(0,))
    t = os.listdir(os.path.join('data','1'))
    for file in t:
        temp = os.path.join('data','1',file)
        ds.addSample(load_image(temp),(1,))
    t = os.listdir(os.path.join('data','2'))
    for file in t:
        temp = os.path.join('data','2',file)
        ds.addSample(load_image(temp),(2,))
    t = os.listdir(os.path.join('data','3'))
    for file in t:
        temp = os.path.join('data','3',file)
        ds.addSample(load_image(temp),(3,))
    t = os.listdir(os.path.join('data','4'))
    for file in t:
        temp = os.path.join('data','4',file)
        ds.addSample(load_image(temp),(4,))
    t = os.listdir(os.path.join('data','5'))
    for file in t:
        temp = os.path.join('data','5',file)
        ds.addSample(load_image(temp),(5,))
    t = os.listdir(os.path.join('data','6'))
    for file in t:
        temp = os.path.join('data','6',file)
        ds.addSample(load_image(temp),(6,))
    t = os.listdir(os.path.join('data','7'))
    for file in t:
        temp = os.path.join('data','7',file)
        ds.addSample(load_image(temp),(7,))
    t = os.listdir(os.path.join('data','8'))
    for file in t:
        temp = os.path.join('data','8',file)
        ds.addSample(load_image(temp),(8,))
    t = os.listdir(os.path.join('data','9'))
    for file in t:
        temp = os.path.join('data','9',file)
        ds.addSample(load_image(temp),(9,))

    """
    ds.addSample(load_image('0.pbm'),(0,))
    ds.addSample(load_image('1.pbm'),(1,))
    ds.addSample(load_image('2.pbm'),(2,))
    ds.addSample(load_image('3.pbm'),(3,))
    ds.addSample(load_image('4.pbm'),(4,))
    ds.addSample(load_image('5.pbm'),(5,))
    ds.addSample(load_image('6.pbm'),(6,))
    ds.addSample(load_image('7.pbm'),(7,))
    ds.addSample(load_image('8.pbm'),(8,))
    ds.addSample(load_image('9.pbm'),(9,))
    """
    return ds

def load_image(path):
    im = cv2.imread(path)
    return vectorize(im)

def vectorize(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(vectorize(el))
        else:
            result.append(el)
    return result

def classify(imSize, dataset, hidden_neurons, initial_error):
    print("Capas de entrada: %i" % imSize) #numero de layouts de entrada, tiende a ser wxh de la imagen
    CLASSES = 10 #10 clases porque son 10 numeros

    #tstdata, trndata = dataset.splitWithProportion( 0.25 )
    #nos da una proporcion de data de entrenamiento de .75 y prueba .25

    #imSize es el tamano de las capas de entrada
    #el siguiente parametro es el de las capas ocultas
    #y el ultimo es las capas de salida que debe haber
    net = buildNetwork(imSize, imSize/3, 1)

    #fnn = buildNetwork(trndata.indim, hidden_neurons, trndata.outdim,
    #                   outclass=SoftmaxLayer)

    #Creamos un entrenador de retropropagacion usando el dataset y la red
    trainer = BackpropTrainer(net, dataset)
    #trainer = BackpropTrainer(fnn, trndata)
    error = initial_error
    iteration = 0
    #iteramos mientras el error sea menor 0.001
    while error > 0.001:
        error = trainer.train()
        iteration += 1
        print "Iteration: {0} Error {1}".format(iteration, error)

    #return fnn
    return net


if __name__ == '__main__':
    #cargar toda la data en una variable alldata
    imSize = len(load_image(os.path.join('data','size.pbm')))
    alldata = load_data(imSize)

    number = classify(imSize,alldata,imSize,10)

    ruta = raw_input("Introduzca ruta del archivo:\n")

    print "Result: ", number.activate(load_image(ruta))
