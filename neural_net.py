import numpy as np
from scipy import optimize

"""
Supervised regression problem
"""

def main(): 
  # Numpy array representing (1) the number of hours of sleep
  # and (2) the number of hours of studying.
  x = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)

  # Numpy array containing the test score on the exam as a
  # result of the sleep/studying habit.
  y = np.array(([75], [82], [93]), dtype=float)


  """
  Before throwing data in the model we need to account for the
  differences in the units of our data (input: (hours, hours),
  output: (test score)) scale result between 0 and 1 by dividing
  both sets by the maximum value possible.
  """

  x = x/np.amax(x, axis=0)
  y = y/100 # Max test score is 100.


  """
  Sample out put from out neural network on first pass. We learn that our neural net is pretty dumb tho. So lets minimize this loss.
  """
  NN = Neural_Network()
  yHat = NN.forward(x)
  print yHat

  NN = Neural_Network()
  cost1 = NN.costFunction(x, y)
  dJdW1, dJdW2 = NN.costFunctionPrime(x, y)
  print cost1

  NN = Neural_Network()
  T = trainer(NN)
  T.train(x, y)
  yHat = NN.forward(x)
  print yHat


"""
Now we can build the neural network. Our network must have two
inputs and one output. We can call out input y-hat because we
only have an estimation of what y is. Any layer between input
and output are called hidden layers; with MANY layers we get
the term deep belief network and hence Deep Learning.
"""

class Neural_Network(object):
  def __init__(self):
    # Define HyperParameters constants that define the structure
    # and behavior of our network but are not updated as the
    # network is used.
    self.inputLayerSize = 2
    self.outputLayerSize = 1
    self.hiddenLayerSize = 3

    # Weights (Parameters)
    self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
    self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

  def forward(self, X):
    # Propagate inputs through the network
    self.z2 = np.dot(X, self.W1)
    self.a2 = self.sigmoid(self.z2)
    self.z3 = np.dot(self.a2, self.W2)
    yHat = self.sigmoid(self.z3)
    return yHat

  def sigmoid(self, z):
    # Apply sigmoid activiation function
    return 1/(1 + np.exp(-z))

  def sigmoidPrime(self, z):
    # Apply sigmoid activiation function
    return np.exp(-z)/((1 + np.exp(-z)) ** 2)
    
  def costFunction(self, X, y):
    #Compute cost for given X,y, use weights already stored in class.
    self.yHat = self.forward(X)
    J = 0.5*sum((y-self.yHat)**2)
    return J

  def costFunctionPrime(self, X, y):
    # Compute derivative with respect to W1 and W2
    self.yHat = self.forward(X)

    delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
    dJdW2 = np.dot(self.a2.T, delta3)

    delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.a2)
    dJdW1 = np.dot(X.T, delta2)

    return dJdW1, dJdW2

  def getParams(self):
    # Get W1 and W2 unrolled into vector:
    params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
    return params

  def setParams(self, params):
    # Set W1 and W2 using single parameter vector.
    W1_start = 0
    W1_end = self.hiddenLayerSize * self.inputLayerSize
    self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
    W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
    self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

  def computeGradients(self, X, y):
    dJdW1, dJdW2 = self.costFunctionPrime(X, y)
    return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

class trainer(object):
  def __init__(self, N):
    # Make local reference to Neural Network
    self.N = N

  def costFunctionWrapper(self, params, X, y):
    self.N.setParams(params)
    cost = self.N.costFunction(X, y)
    grad = self.N.computeGradients(X, y)
    return cost, grad

  def callbackF(self, params):
    self.N.setParams(params)
    self.J.append(self.N.costFunction(self.X, self.y))

  def train(self, X, y):
    # Make internal variable for callback function:
    self.X = X
    self.y = y

    # Make empty list to store costs:
    self.J = []

    params0 = self.N.getParams()

    options = {'maxiter': 200, 'disp': True}
    _res = optimize.minimize(self.costFunctionWrapper, params0, jac = True, method='BFGS', args = (X,y), options=options, callback=self.callbackF)

    self.N.setParams(_res.x)
    self.optimizationResults = _res

def computeNumericalGradient(N, X, y):
  paramsInitial = N.getParams()
  numgrad = np.zeros(paramsInitial.shape)
  perturb = np.zeros(paramsInitial.shape)
  e = 1e-4

  for p in range(len(paramsInitial)):
    # Set perturbation vector
    perturb[p] = e
    N.setParams(paramsInitial + perturb)
    loss2 = N.costFunction(X, y)

    N.setParams(paramsInitial - perturb)
    loss1 = N.costFunction(X, y)

    # Compute Numerical Gradient
    numgrad[p] = (loss2 - loss1) / (2 * e)

    # Return the value we changed to zero:
    perturb[p] = 0

  # Return Params to original value
  N.setParams(paramsInitial)

  return numgrad

if __name__ == '__main__':
  main()
