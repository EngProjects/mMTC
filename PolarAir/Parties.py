# -*- coding: utf-8 -*-
import numpy as np
import torch.nn as nn
import torch
from datetime import datetime
import os
import pathlib


# ======================================== WORKER ======================================== #

class Worker:
    # --- Constructor
    def __init__(self, K, nPar, device, dtype):
        self.K = K  # Number of non-zero elements
        self.nPar = nPar  # Number of the parameters of the CNN
        self.device = device  # to store the device (GPU only)
        self.dtype = dtype  # to store the type
        self.Delta = np.zeros(nPar)  # To store the Error

    def flatGradient(self, model):

        """ Function to flat the gradient CNN
            Inputs:
                    1. model
            Outputs:
                    The flat Gradient ( vector of dimensions self.nPar)
        """

        # --- To store the parameters
        grads = []

        # --- Iterate over all parameters
        for param in model.parameters():
            grads.append(param.grad.view(-1))

        # --- Concatenates the given sequence of tensors
        grads = torch.cat(grads)

        # --- Return the flat Gradient
        return grads

    def train(self, x, y, model, optimizer):
        """ Function to train the CNN
            Inputs:
                    1. x, y: current batch
                    2. model: ResNet20
                    3. optimizer: Adam
            Outputs:
                    1. top-K gradient
                    2. The set of active indices
        """

        # --- Send the model to GPU
        model = model.to(device=self.device)

        # --- Put model to training mode
        model.train()

        # --- Send the parameters to GPU
        xBatch = x.to(device=self.device, dtype=self.dtype)
        yBatch = y.to(device=self.device, dtype=torch.long)

        # --- Output of the NN
        scores = model(xBatch)

        # --- Compute the Loss
        loss = nn.functional.cross_entropy(scores, yBatch)

        # --- Zero the gradient
        optimizer.zero_grad() # Set the gradients to zero to compute the next gradient

        # --- Compute the gradient of the loss with respect to each parameter of the model
        loss.backward()

        # --- Make the gradient Flat
        gradient = self.flatGradient(model)

        # --- Error Accumulation Step
        if torch.cuda.is_available():
            gEA = gradient.cpu().numpy() + self.Delta
        else:
            gEA = gradient + self.Delta

        # --- Sparsify the Gradient
        gK, idx = sparGradient(gEA, self.K, self.nPar)

        # --- Compute the error
        self.Delta = gEA - gK

        return gK, idx


def sparGradient(gradient, K, nPar):
    """ Function to sparsify the gradient
            Inputs:
                    1. gradient: dense gradient
                    2. K: number of non-zero values
                    3. nPar: number of parameters
            Outputs:
                    1. The sparse gradient
                    2. The set of active indices
    """

    # --- Create a vector to store the active indices
    gK = np.zeros(nPar)

    # --- Fint the greatest in absolute values in gradient
    idx = (-abs(gradient)).argsort()[:K]

    # --- Store the active indices
    gK[idx] = gradient[idx]

    return gK, idx


# ======================================== SERVER ======================================== #

class Server:
    # --- Constructor
    def __init__(self, K, nPar, device, dtype, model, optimizer):
        self.K = K  # Number of non-zero elements
        self.nPar = nPar  # Number of the parameters of the CNN
        self.device = device  # to store the device (GPU only)
        self.dtype = dtype  # to store the type
        self.model = model  # Store the model
        self.optimizer = optimizer  # Adam

    # --- Reshape Gradient and pass it to the model
    def changeGradient(self, gradient):

        """ Function to save the recovered gradient before the gradient descent step
                Inputs:
                        1. gradient: recovered gradient
                Resulting Output:
                        1. Model ready to be optimized
        """
        count = 0

        # --- Iterate over all model's parameters
        for par in self.model.parameters():
            # --- Find the shape of the current parameters
            shape = par.shape

            # --- Length of current parameters
            length = prod(shape)

            # --- Reshape
            gradTemp = np.reshape(gradient[count:count + length], shape)
            gradTemp = torch.from_numpy(gradTemp)

            # --- Pass it to the model
            par.grad = (gradTemp.to(torch.float32)).cuda()

            count += length

    def step(self, gradient):

        """ Function to take the Gradient Descent Step
                Inputs:
                        1. gradient: recovered gradient
                        2. x,y: test data if we want to check the performance
                Resulting Output:
                        1. Updated model
        """

        # --- First save the current values of the gradient to the model
        self.changeGradient(gradient)

        # Update the parameters of the model using the gradients computed by the backwards pass
        self.optimizer.step()


    def checkAcc(self, X, Y, model, device, dtype):

        numCorr, numSamples = 0, 0
        model.eval()  # set the model to evaluation mode

        with torch.no_grad():
                x = X.to(device=device, dtype=dtype)
                y = Y.to(device=device, dtype=dtype)

                # Output of the NN
                scores = model(x)

                # Select the largest value as the ouput
                _, predictions = scores.max(1)

                # Count the number of correct decisions
                numCorr += (predictions == y).sum()

                # Count the number of data that pass through the network
                numSamples = X.shape[0]

                # Compute the accuracy
                acc = float(numCorr) / numSamples
                print('Got %d / %d correct (%.2f)' % (numCorr, numSamples, 100 * acc))
                return acc


def prod(shape):
    length = 1
    for i in shape:
        length *= i
    return length


class Experimentalist:

    # --- Constructor
    def __init__(self, K, dB, W, nPar, C, epochs, POLAR_SCHEME, NOISE):
        self.K = K
        self.dB = dB
        self.W = W
        self.nPar = nPar
        self.POLAR_SCHEME = POLAR_SCHEME
        self.NOISE = NOISE
        # Initialize matrices to save the results
        self.testError = np.zeros(epochs)
        self.falseAlarm = np.zeros((C, epochs))
        self.deInFA = np.zeros((C, epochs))
        self.detection = np.zeros((C, epochs))
        self.mse = np.zeros((C, epochs))
        self.activeIdx = np.storeAbs = np.zeros((C, epochs))
        self.maxRecovered = np.zeros((C, epochs))
        self.minRecovered = np.zeros((C, epochs))
        self.maxOriginal = np.zeros((C, epochs))
        self.minOriginal = np.zeros((C, epochs))
        self.decoderIter = np.zeros((C, epochs))
        self.numIdxHat = np.zeros((C, epochs))
        self.numOfMeas = np.zeros((C, epochs))
        self.snrdB = np.zeros((C, epochs))

    def store(self, c, i, numIdx, gradientHat, idxHat, idx, originalGrad, iterations, s):
        # number of recovered indices
        self.numIdxHat[c, i] = numIdx
        # Probabilitu=y of false alarm
        self.falseAlarm[c, i] = self.falseAlarm[c, i] / float(numIdx)
        # Probability of Detection
        self.detection[c, i] = self.detection[c, i] / float(self.K)
        # Mean Square Error
        tempGr = np.zeros(self.nPar)
        tempGr[idx] = originalGrad[idx] / float(self.W)
        self.mse[c, i] = np.sum((tempGr - gradientHat) ** 2) / np.sum((gradientHat ** 2))
        # Max value of the recovered gradient
        self.maxRecovered[c, i] = max(abs(gradientHat[idxHat]))
        # Min value of the recovered gradient
        self.minRecovered[c, i] = min(abs(gradientHat[idxHat]))
        # Max value of the actual gradient
        self.maxOriginal[c, i] = max(abs(originalGrad[idx]))
        # Min value of the actual gradient
        self.minOriginal[c, i] = min(abs(originalGrad[idx]))
        # Number of iterations
        self.decoderIter[c, i] = iterations
        # Number of measuments
        self.numOfMeas[c, i] = s

    def saveToCsv(self):

        # Find date and time
        now = datetime.now()
        dateTime = now.strftime("%d_%m_%Y-%H:%M")

        # Create the path to create a directory
        path = os.path.join(pathlib.Path().resolve(), now.strftime("%d-%m-%Y_%H-%M"))

        # Create a directory
        os.mkdir(path)
        # Move to that directory
        os.chdir(path)

        # Save

        if self.POLAR_SCHEME and self.NOISE:
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_deInFA.csv", self.deInFA, delimiter=",")
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_activeIdx.csv", self.activeIdx, delimiter=",")
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_Q.csv", self.numIdxHat, delimiter=",")
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_falseAlarm.csv", self.falseAlarm, delimiter=",")
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_detection.csv", self.detection, delimiter=",")
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_mse.csv", self.mse, delimiter=",")
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_maxRecovered.csv", self.maxRecovered,
                       delimiter=",")
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_minRecovered.csv", self.minRecovered,
                       delimiter=",")
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_maxOriginal.csv", self.maxOriginal,
                       delimiter=",")
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_minOriginal.csv", self.minOriginal,
                       delimiter=",")
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_decoderIter.csv", self.decoderIter,
                       delimiter=",")
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_snr.csv", self.snrdB / float(self.W),
                       delimiter=",")
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_numOfMeas.csv", self.numOfMeas, delimiter=",")
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_testAccu.csv", self.testError, delimiter=",")


        elif self.POLAR_SCHEME and (not self.NOISE):
            np.savetxt("K" + str(self.K) + "Noise" + str(self.dB) + "dB_Q.csv", self.numIdxHat, delimiter=",")
            np.savetxt("K" + str(self.K) + "falseAlarm.csv", self.falseAlarm, delimiter=",")
            np.savetxt("K" + str(self.K) + "detection.csv", self.detection, delimiter=",")
            np.savetxt("K" + str(self.K) + "mse.csv", self.mse, delimiter=",")
            np.savetxt("K" + str(self.K) + "maxRecovered.csv", self.maxRecovered, delimiter=",")
            np.savetxt("K" + str(self.K) + "minRecovered.csv", self.minRecovered, delimiter=",")
            np.savetxt("K" + str(self.K) + "maxOriginal.csv", self.maxOriginal, delimiter=",")
            np.savetxt("K" + str(self.K) + "minOriginal.csv", self.minOriginal, delimiter=",")
            np.savetxt("K" + str(self.K) + "decoderIter.csv", self.decoderIter, delimiter=",")
            np.savetxt("K" + str(self.K) + "snr.csv", self.snrdB / float(self.W), delimiter=",")
            np.savetxt("K" + str(self.K) + "numOfMeas.csv", self.numOfMeas, delimiter=",")
            np.savetxt("K" + str(self.K) + "testError.csv", self.testError, delimiter=",")

        else:
            np.savetxt("K" + str(self.K) + "testError.csv", self.testError, delimiter=",")



