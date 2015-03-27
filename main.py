#!/usr/bin/env python
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError

def load_data(route):
    return

def classify(training, testing, HIDDEN_NEURONS, MOMENTUM, WEIGHTDECAY,
             LEARNING_RATE, LEARNING_RATE_DECAY, EPOCHS):
    INPUT_FEATURES = 0 #numero de layouts de entrada, tiende a ser wxh de la imagen
    print("Input features: %i" % INPUT_FEATURES)
    CLASSES = 10 #10 clases porque son 10 numeros
    trndata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    tstdata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)

    #en este segmento se agregan los samples de training y test
    for i in range(len(testing['x'])):
        tstdata.addSample(ravel(testing['x'][i]), [testing['y'][i]])
    for i in range(len(training['x'])):
        trndata.addSample(ravel(training['x'][i]), [training['y'][i]])

    # This is necessary, but I don't know why
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()

    fnn = buildNetwork(trndata.indim, HIDDEN_NEURONS, trndata.outdim,
                       outclass=SoftmaxLayer)

    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=MOMENTUM,
                              verbose=True, weightdecay=WEIGHTDECAY,
                              learningrate=LEARNING_RATE,
                              lrdecay=LEARNING_RATE_DECAY)
    for i in range(EPOCHS):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(),
                                 trndata['class'])
        tstresult = percentError(trainer.testOnClassData(
                                 dataset=tstdata), tstdata['class'])
        #El ciclo deberia detenerse al obtener <=95%
        print("epoch: %4d" % trainer.totalepochs,
                     "  train error: %5.2f%%" % trnresult,
                     "  test error: %5.2f%%" % tstresult)
    return fnn


if __name__ == '__main__':
    ruta = raw_input("Introduzca ruta del archivo:\n")
    print ruta
    ruta_test = ""
    ruta_training = ""
    testing = load_data(ruta_test)
    training = load_data(ruta_training)
    imagen = load_data(ruta)
    #classify()
    #compare()

"""
#primitivas a usar probablemente
n = FeedForwardNetwork()
inLayer = LinearLayer(2)
hiddenLayer = SigmoidLayer(3)
outLayer = LinearLayer(1)
n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

#lo malo de esto es que no se contempla la subida de archivos

n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)

n.sortModules()
print n

n.activate([1, 2])
"""
