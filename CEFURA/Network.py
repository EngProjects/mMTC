import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self, D, nAPs, nAnts, nUEs, type, densityP, densityD, radius, sigma2P):
        self.D = D
        self.nAPs = nAPs # Number of Access Points (AP)
        self.nAnts = nAnts # Number of antennas per AP
        self.nUEs = nUEs # Number of user equipment (UE)
        self.ha = 15 # Height of AP
        self.hu = 1.65 # Height of UE
        self.L = -30.5
        self.computeCentroids()
        self.type = type
        self.densityP = densityP
        self.densityD =densityD
        self.radius = radius
        self.sigma2P = sigma2P

    def computeCentroids(self):

        # --- Find the number of AP in a row
        self.nAPsPerRow = int(np.sqrt(self.nAPs))

        # --- Find the number of AP in a col
        self.nAPsPerCol = self.nAPsPerRow


        # --- Compute the area
        self.area = self.D * self.D

        # --- Compute the area for each AP
        self.areaForAP = self.area / self.nAPs

        # --- Compute the length of the square edge
        lenght = np.sqrt(self.areaForAP)

        step = lenght

        # --- Collect all the centroids
        self.APsCoordinates = np.zeros((2, self.nAPs))
        self.APsCoordinates[0, 0] = lenght / 2
        self.APsCoordinates[1, 0] = lenght / 2
        idx = 0
        for r in range(self.nAPsPerRow - 1, -1, -1):
            for c in range(self.nAPsPerCol):
                self.APsCoordinates[0, idx] = (self.nAPsPerRow - 1 - r) * step + (lenght / 2)
                self.APsCoordinates[1, idx] = step * c + (lenght / 2)
                idx += 1


    def plotCell(self):

        plt.scatter(self.APsCoordinates[0, :], self.APsCoordinates[1, :])

        if True and self.UEsCoordinates[0,0] != 0:
            plt.scatter(self.UEsCoordinates[0, :], self.UEsCoordinates[1, :])
            plt.legend(['APs', 'UEs'], loc='upper right')
            for i, txt in enumerate(np.arange(self.nUEs)):
                plt.annotate(txt, (self.UEsCoordinates[0, i], self.UEsCoordinates[1, i]))
        else:
            plt.legend(['APs'], loc='upper right')

        for i, txt in enumerate(np.arange(self.nAPs)):
            plt.annotate(txt, (self.APsCoordinates[0, i], self.APsCoordinates[1, i]))

        plt.grid()
        plt.xlim([0, self.D])
        plt.ylim([0, self.D])
        plt.xlabel('x-Axis')
        plt.ylabel('y-Axis')
        plt.show()


    def generateUsers(self):
        self.UEsCoordinates, self.nUEs = clusterProcess(self.type , 0, 0, self.D, self.D, self.nUEs, \
                                                            self.densityP, self.densityD, self.radius, self.sigma2P)


        self.distUE2AP = np.zeros((self.nUEs, self.nAPs))
        self.distUE2UE = np.zeros((self.nUEs, self.nUEs))
        self.pathLoss = np.zeros((self.nUEs, self.nAPs))
        self.shadowingCovUEs = np.zeros((self.nUEs, self.nUEs))

        self.computeLargeScaleCoeff()

    def computeLargeScaleCoeff(self):

        # === From APs to UEs
        for ap in range(self.nAPs):

            pointA = np.append(self.APsCoordinates[:,ap], self.ha)

            for ue in range(self.nUEs):
                pointB = np.append(self.UEsCoordinates[:,ue], self.hu)

                # --- Compute the Distance
                self.distUE2AP[ue, ap] = distance(pointA, pointB)

                # --- Compute the Path Loss (large scale coefficient)
                self.pathLoss[ue, ap] = np.sqrt(10 ** (self.computePathLossdB(self.distUE2AP[ue, ap])/10))

        self.contructShadowingCov()



    def contructShadowingCov(self):
        shadowingCovUEsTemp = np.zeros((self.nUEs, self.nUEs))
        # === From UEs to UEs
        for ueR in range(self.nUEs):
            pointA = np.append(self.UEsCoordinates[:, ueR], self.hu)

            for ueC in range(ueR, self.nUEs, 1):

                pointB = np.append(self.UEsCoordinates[:, ueC], self.hu)

                if ueC != ueR:
                    # Compute distance
                    self.distUE2UE[ueR, ueC] = distance(pointA, pointB)

                    # Compute joint Expectation
                    shadowingCovUEsTemp[ueR, ueC] = 16 * (2 ** (-self.distUE2UE[ueR, ueC]/9))
                else:
                    shadowingCovUEsTemp[ueR, ueC] = 8

        temp = shadowingCovUEsTemp + shadowingCovUEsTemp.T
        self.shadowingCovUEs = temp

        # Find Eigenvalue decomposition of the covariance matrix
        S, U = np.linalg.eig(self.shadowingCovUEs)

        self.WUEs = np.dot(U, np.sqrt(np.diag(S)))




    def computePathLossdB(self, distance):

        pathLossTemp = -36.7*np.log10(distance)
        return self.L + pathLossTemp

    def generateChannel(self):

        # --- Small scale
        self.h = np.sqrt(0.5) * (np.random.normal(0,1, (self.nUEs, self.nAPs * self.nAnts)) + 1j*np.random.normal(0,1, (self.nUEs, self.nAPs * self.nAnts)))

        # --- Large Scale
        # Shadowing
        shandowing = computeShadowing(self)

        self.beta05 = self.pathLoss * np.sqrt(10 ** (shandowing/10))

        beta05Temp = np.matlib.repmat(self.beta05, 1, self.nAnts)
        beta05Temp = np.reshape(beta05Temp, (self.nUEs, self.nAnts, self.nAPs))
        self.beta05 = np.reshape(beta05Temp.T, (self.nAnts * self.nAPs, self.nUEs)).T

        self.H = self.h * self.beta05

        # return self.h
        return self.H

def computeShadowing(self):

    z = np.random.normal(loc=0.0, scale=1, size=(self.nUEs, self.nAPs))
    F = np.dot(self.WUEs, z)
    return F

def distance(pointA, pointB):
    return np.sqrt(np.sum((pointA - pointB)**2))



def clusterProcess(type, xMin, yMin, xMax, yMax,  nUEs, densityP, densityD, radius, sigma2):


    if type == 'B':
        # Binomial Point Process (BPP) i.e. Conditional PPP
        return np.vstack((np.random.uniform(xMin, xMax, nUEs), np.random.uniform(yMin, yMax, nUEs))), nUEs
    else:

        # === Generate Thomas Cluster Process (TCP) or Matérn Cluster Point (MCP)
        # The only difference between these processes is how the points inside the
        # clusters are generated

        # === Common Steps ==== #

        # ---- Step 0: Pre-processing
        # Extended simulation rectangle parameters
        extra = (type == 'T') * (6 * np.sqrt(sigma2)) + (type == 'M') * radius
        xMinExt = xMin - extra
        yMinExt = yMin - extra
        xMaxExt = xMax + extra
        yMaxExt = yMax + extra

        # Calculate Area
        area = (xMaxExt - xMinExt) * (yMaxExt - yMinExt)


        # ---- Step 1: Parent Process
        # Generate the number of points clusters (Poisson)
        nParents = np.random.poisson(densityP * area)

        # Generate the locations (Poisson)
        xParents = xMinExt + (xMaxExt - xMinExt) * np.random.uniform(0, 1, nParents)
        yParents = yMinExt + (yMaxExt - yMinExt) * np.random.uniform(0, 1, nParents)


        # ---- Step 2: Daughter Process
        # Generate the number of points within the cluster (Poisson)
        nDaughterPerCluster = np.random.poisson(densityD, nParents)
        nPoints = sum(nDaughterPerCluster)

        if type == 'P':
            return np.vstack((xParents,yParents)), len(yParents)
        elif type == 'T':
            # Generate the locations (Gaussian)
            xPoints = np.random.normal(0, np.sqrt(sigma2), nPoints)
            yPoints = np.random.normal(0, np.sqrt(sigma2), nPoints)

        elif type == 'M':

            # Generate the polar coordinates
            theta = 2 * np.pi * np.random.uniform(0, 1, nPoints)
            rho = radius * np.sqrt(np.random.uniform(0, 1, nPoints))

            # Convert from polar to Cartesian coordinates
            xPoints = rho * np.cos(theta)
            yPoints = rho * np.sin(theta)


        # --- Step 3: Shift all the points to their clusters
        # Pre-processing
        xParents = np.repeat(xParents, nDaughterPerCluster)
        yParents = np.repeat(yParents, nDaughterPerCluster)

        # Shift the points
        xPoints = xParents + xPoints
        yPoints = yParents + yPoints

        keep = ((xPoints >= xMin) & (xPoints <= xMax) & (yPoints >= yMin) & (yPoints <= yMax))


        return np.vstack((xPoints[keep], yPoints[keep])), len(yPoints[keep])


