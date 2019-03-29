import sys
import random
import numpy as np
from copy import deepcopy

import qml
from qml.representations import generate_bob
from qml.kernels import gaussian_kernel
from qml.kernels import laplacian_kernel
from qml.math import cho_solve

def get_properties(filename):
  """ returns dict with properties for xyz files
  """
  f = open(filename, "r")
  lines = f.readlines()
  f.close()

  properties = dict()

  for line in lines:
    tokens = line.split()
    xyz_name = tokens[0]
    prop = float(tokens[1])*0.239
    properties[xyz_name] = prop

  return properties 

def get_coords(filename):
  """ returns dict for xyz files
  """
  f = open(filename, "r")
  lines = f.readlines()
  f.close()

  coords = dict()

  for line in lines:
    tokens = line.split()
    coord = tokens[0]
    coords[coord] = 0

  return coords

def get_representation(data, data2, path_to_xyz_files):
  """ calculates the representations and stores it in the qml compound class
  """
  mols  = []
  mols2 = []

  for xyz_file in sorted(data.keys()):
    mol = qml.data.Compound()
    mol.read_xyz(path_to_xyz_files + xyz_file + ".xyz")
    mol.properties = data[xyz_file]
    mols.append(mol)

  for xyz_file in sorted(data2.keys()):
    mol = qml.data.Compound()
    mol.read_xyz(path_to_xyz_files + xyz_file + ".xyz")
    mol.properties = data2[xyz_file]
    mols2.append(mol)

  bags = {
    "H":  max([mol.atomtypes.count("H" ) for mol in mols+mols2]),
    "C":  max([mol.atomtypes.count("C" ) for mol in mols+mols2]),
    "N":  max([mol.atomtypes.count("N" ) for mol in mols+mols2]),
    "O":  max([mol.atomtypes.count("O" ) for mol in mols+mols2]),
    "S":  max([mol.atomtypes.count("S" ) for mol in mols+mols2]),
    "F":  max([mol.atomtypes.count("F" ) for mol in mols+mols2]),
    "Si": max([mol.atomtypes.count("Si") for mol in mols+mols2]),
    "P":  max([mol.atomtypes.count("P" ) for mol in mols+mols2]),
    "Cl": max([mol.atomtypes.count("Cl") for mol in mols+mols2]),
    "Br": max([mol.atomtypes.count("Br") for mol in mols+mols2]),
    "Ni": max([mol.atomtypes.count("Ni") for mol in mols+mols2]),
    "Pt": max([mol.atomtypes.count("Pt") for mol in mols+mols2]),
    "Pd": max([mol.atomtypes.count("Pd") for mol in mols+mols2]),
    "Cu": max([mol.atomtypes.count("Cu") for mol in mols+mols2]),
    "Ag": max([mol.atomtypes.count("Ag") for mol in mols+mols2]),
    "Au": max([mol.atomtypes.count("Au") for mol in mols+mols2])
  }

  for mol in mols:
    mol.generate_bob(asize=bags)

  for mol in mols2:
    mol.generate_bob(asize=bags)

  return mols, mols2

def cross_validation(X, Y, sigmas, llambdas, Ntot):
  """ finds optimal hyperparameters sigma & lambda using cross validation
  """
  parameters = []
  random.seed(666)

  for i in range(len(sigmas)):
    K = laplacian_kernel(X, X, sigmas[i])

    for j in range(len(llambdas)):
  
      for m in range(5):
        maes = []
        split = range(Ntot)
        random.shuffle(split)

        train = int(len(split)*0.8)
        test  = int(Ntot - train)

        training_index  = split[:train]
        test_index      = split[-test:]

        y_train = Y[training_index]
        y_test  = Y[test_index]

        C = deepcopy(K[training_index][:,training_index])
        C[np.diag_indices_from(C)] += llambdas[j]

        alpha = cho_solve(C, y_train)

        y_est = np.dot((K[training_index][:,test_index]).T, alpha)

        diff = y_est  - y_test
        mae = np.mean(np.abs(diff))
        maes.append(mae)

      parameters.append([llambdas[j], sigmas[i], np.mean(maes)])

  maes = [mae[2] for mae in parameters]
  index = maes.index(min(maes))

  print("minimum MAE after CV: ", min(maes))

  return parameters[index][0], parameters[index][1]

def get_alphas(X, Y, sigma, llambda):
  ''' calculates the regression coefficient alpha
  '''
  K = laplacian_kernel(X, X, sigma)

  C = deepcopy(K)
  C[np.diag_indices_from(C)] += llambda

  alpha = cho_solve(C, Y)

  return alpha

def get_predictions(mols_pred, X, X_pred, alpha, sigma):
  ''' predicts proerties
  '''
  K_pred = laplacian_kernel(X, X_pred, sigma)

  Yss = np.dot(K_pred.T, alpha)

  for i in range(len(Yss)):
    print(str(mols_pred[i].name) + "\t" + str(Yss[i]))

def get_learning_curve(X, X_test, Y, Y_test, sigma, llambda, Ntot):
  ''' generate data (predictions) for learning curves
  '''
  K			 = laplacian_kernel(X, X,      sigma)
  K_test = laplacian_kernel(X, X_test, sigma)

  N = []
  j = 10

  while(j < Ntot):
    N.append(j)
    j *= 2 

  N.append(Ntot)

  random.seed(667)

  for train in N:
    maes = []

    for i in range(10):
      split = range(Ntot)
      random.shuffle(split)

      training_index = split[:train]

      y = Y[training_index]

      C = deepcopy(K[training_index][:,training_index])
      C[np.diag_indices_from(C)] += llambda 
                                                 
      alpha = cho_solve(C, y)                          

      Yss = np.dot(K_test[training_index].T, alpha)

      diff = Yss - Y_test
      mae = np.mean(np.abs(diff))
      maes.append(mae)

    print(str(train) + "\t" + str(sum(maes)/len(maes)))

def LC(training_data_file, test_pred_data_file, path_to_xyz_files):
  ''' calls all necessary functions to generate the data for learning curves
  '''
  training_data 	= get_properties(training_data_file)
  test_data				= get_properties(test_pred_data_file)

  mols, mols_test = get_representation(training_data, test_data, path_to_xyz_files)

  X      = np.asarray([mol.representation for mol in mols])
  Y      = np.asarray([mol.properties for mol in mols])

  X_test = np.asarray([mol.representation for mol in mols_test])
  Y_test = np.asarray([mol.properties for mol in mols_test])

  Ntot = len(training_data)
  llambdas = [ 1e-3, 1e-5, 1e-7, 1e-9]
  sigmas  = [0.1*2**i for i in range(5,20)]

  llambda, sigma = cross_validation(X, Y, sigmas, llambdas, Ntot)

  print(llambda, sigma)
  get_learning_curve(X, X_test, Y, Y_test, sigma, llambda, Ntot)

def predictions(training_data_file, test_pred_data_file, path_to_xyz_files):
  ''' calls all necessary functions to do the predictions 
  '''
  training_data 			= get_properties(training_data_file)
  prediction_data 		= get_coords(test_pred_data_file)
  mols, mols_predict 	= get_representation(training_data, prediction_data, path_to_xyz_files)

  X      = np.asarray([mol.representation for mol in mols])
  Y      = np.asarray([mol.properties for mol in mols])

  X_pred = np.asarray([mol.representation for mol in mols_predict])

  Ntot = len(training_data)
  llambdas = [ 1e-3, 1e-5, 1e-7, 1e-9]
  sigmas  = [0.1*2**i for i in range(5,20)]

  llambda, sigma = cross_validation(X, Y, sigmas, llambdas, Ntot)
  alpha = get_alphas(X, Y, sigma, llambda)

  print(llambda, sigma)
  get_predictions(mols_predict, X, X_pred, alpha, sigma)

if __name__ == "__main__":

  training_data_file 		= sys.argv[1]
  test_pred_data_file  = sys.argv[2]
  path_to_xyz_files  		= sys.argv[3]
  decision 							= sys.argv[4]

  if decision == "learning_curves":
    LC(training_data_file, test_pred_data_file, path_to_xyz_files)

  elif decision == "predictions":
    predictions(training_data_file, test_pred_data_file, path_to_xyz_files)

  else:
    print("no valid decicion, please choose between <learning_curves> and <predictions>\n python BoB.py <train.txt> <test.txt or pred.txt> <path to xyz files> <learning_curves or predictions>")

