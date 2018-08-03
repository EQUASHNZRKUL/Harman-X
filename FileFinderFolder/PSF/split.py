from random import seed, choice
from os import listdir, system
import numpy as np

def main(src_dir, biasdict, target_dir):
  """ Categorizes datapoints in [src_dir] according to the weights in [biasdict] where
  the keys are titles of the folders to copy the data into and the values are
  the weights. [target_dir] is the directory to put the folders and the copied data.

  Meant to be used to separate data into different sections. 
  """
  # Clears existing folders
  system("rm " + target_dir)
  system("mkdir " + target_dir)
  # f = lambda y: system("mkdir" + target_dir + "./" + y)
  # map(f, biasdict)

  # Unrolls the biasedlist & wipes the biasdict
  seed(1)
  biasedlist = []
  for k, v in biasdict.iteritems():
    biasedlist = biasedlist + [k] * v
    biasdict[k] = {}

  # Reveals the [d] master dict & processes categorization 
  d = np.load(src_dir)
  for cmd, mfcc_array in d.iteritems():
    for mfcc in mfcc_array:
      classification = choice(biasedlist)
      try:
        biasdict[classification][cmd] = np.append(biasdict[classification][cmd], [mfcc], axis=0)
      except KeyError:
        biasdict[classification][cmd] = np.array([mfcc])

  # stores new biasdict
  for k, v in biasdict.iteritems():
    np.savez(target_dir + k, **v)

if __name__ == "__main__":
  bd = {'test':1, 'train':4}
  main("./MFCCData_folder/MFCCData.npz", bd, "./MFCCData_split/")