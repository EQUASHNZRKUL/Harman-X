import os
import VGG
import tensorflow as tf

# Global Vars
FILE_LENGTH = 1568
NUM_COMMANDS = 6

def read_data(directory):
  