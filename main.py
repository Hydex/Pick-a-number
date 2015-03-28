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
    #10 clases porque son 10 numeros
    ds = ClassificationDataSet(size, 1, nb_classes=10,class_labels=['0','1','2','3','4','5','6','7','8','9'])
    #Agregando todos los elementos 0
    t = os.listdir(os.path.join('data','0'))
    for file in t:
        temp = os.path.join('data','0',file)
        ds.addSample(load_image(temp),[0])
    t = os.listdir(os.path.join('data','1'))
    for file in t:
        temp = os.path.join('data','1',file)
        ds.addSample(load_image(temp),[1])
    t = os.listdir(os.path.join('data','2'))
    for file in t:
        temp = os.path.join('data','2',file)
        ds.addSample(load_image(temp),[2])
    t = os.listdir(os.path.join('data','3'))
    for file in t:
        temp = os.path.join('data','3',file)
        ds.addSample(load_image(temp),[3])
    t = os.listdir(os.path.join('data','4'))
    for file in t:
        temp = os.path.join('data','4',file)
        ds.addSample(load_image(temp),[4])
    t = os.listdir(os.path.join('data','5'))
    for file in t:
        temp = os.path.join('data','5',file)
        ds.addSample(load_image(temp),[5])
    t = os.listdir(os.path.join('data','6'))
    for file in t:
        temp = os.path.join('data','6',file)
        ds.addSample(load_image(temp),[6])
    t = os.listdir(os.path.join('data','7'))
    for file in t:
        temp = os.path.join('data','7',file)
        ds.addSample(load_image(temp),[7])
    t = os.listdir(os.path.join('data','8'))
    for file in t:
        temp = os.path.join('data','8',file)
        ds.addSample(load_image(temp),[8])
    t = os.listdir(os.path.join('data','9'))
    for file in t:
        temp = os.path.join('data','9',file)
        ds.addSample(load_image(temp),[9])
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

    tstdata, trndata = dataset.splitWithProportion( 0.25 )
    # nos da una proporcion de data de entrenamiento de .75 y prueba .25

    # imSize es el tamano de las capas de entrada
    # define layer structures
    inLayer = LinearLayer(imSize)
    hiddenLayer = SigmoidLayer(imSize/3)
    outLayer = SoftmaxLayer(1)

    # add layers to network
    net = FeedForwardNetwork()
    net.addInputModule(inLayer)
    net.addModule(hiddenLayer)
    net.addOutputModule(outLayer)

    # define conncections for network
    theta1 = FullConnection(inLayer, hiddenLayer)
    theta2 = FullConnection(hiddenLayer, outLayer)

    # add connections to network
    net.addConnection(theta1)
    net.addConnection(theta2)

    # sort module
    net.sortModules()

    dataset._convertToOneOfMany( )

    fnn = buildNetwork( dataset.indim, imSize/3, dataset.outdim, outclass=SoftmaxLayer )

    #Creamos un entrenador de retropropagacion usando el dataset y la red
    trainer = BackpropTrainer(fnn, dataset)

    error = initial_error
    iteration = 0
    #iteramos mientras el error sea menor 0.001
    while error > 0.01:
        error = trainer.train()
        iteration += 1
        #print "Iteration: {0} Error {1}".format(iteration, error)
    print "Terminado luego de: ",iteration," iteraciones"
    print "Con un error de: ",error
    return fnn

if __name__ == '__main__':
    #cargar toda la data en una variable alldata
    imSize = len(load_image(os.path.join('data','size.pbm')))
    print "Cargando datos..."
    alldata = load_data(imSize)
    print "Entrenando red..."
    final_net = classify(imSize,alldata,imSize,10)
    classes = ['0','1','2','3','4','5','6','7','8','9']
    while True:
        ruta = raw_input("Introduzca ruta del archivo:\n")
        #print type(final_net.activate(load_image(ruta)))
        #print final_net.activate(load_image(ruta))
        resultado = final_net.activate(load_image(ruta))
        elemento = max(resultado)
        #print "Result: ", max(final_net.activate(load_image(ruta)))
        iter = 0
        #class_index = max(xrange(len(resultado)), key=resultado.__getitem__)
        #class_name = classes[class_index]
        #print class_name
        for i in resultado:
            if i==elemento:
                print "Prediccion: ",iter
            iter+=1
        if raw_input("\n1:Seguir-probando  __ OtroSimbolo:Parar ::")!="1":
          break
        print("\n")