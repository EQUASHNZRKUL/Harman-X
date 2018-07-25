from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from pydub import AudioSegment
import os

import scipy.io.wavfile as wav
import soundfile as sf
import numpy as np

def get_info(infostr):
  """ Returns tuple of metadata from ami_metadata file. [infostr] is one line
  of an ami_metadata file. Tuple is in format of cmd * starttime * endtime."""
  print infostr
  i1 = infostr.find("id = ")
  i2 = infostr.find("starttime = ")
  i3 = infostr.find("endtime = ")
  id = infostr[i1+5:i2-2]
  st = float(infostr[i2+12:i3-2])
  et = float(infostr[i3+10:-4])
  return (id, st, et)

def cut_ami(filename):
  resfile = open(filename, 'r')
  os.system("rm ../results/cut_ami_results.txt")
  f = open("../results/cut_ami_results.txt", "a")
  key = None
  for line in resfile:
    if "[" in line : 
      key = line[1:-5]
      f.write(line)
    elif (not "[" in line) and (not "]" in line):
      info = get_info(line)
      id = info[0]
      s1 = id.find('.')
      id = id[:s1]
      dir = "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/F\
ileFinderData/AMI/Arrays/Array1-01/" + id + "/audio/" + id + ".Array1-01.wav"
      print dir
      cmd1 = "mkdir /Users/justinkae/Documents/TensorFlowPractice/FileFinderFol\
der/FileFinderData/AMI_cut/Arrays/Array1-01/" + id
      os.system(cmd1)
      cmd2 = cmd1 + "/" + key
      os.system(cmd2)
      i = len(os.listdir("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFol\
der/FileFinderData/AMI_cut/Arrays/Array1-01/" + id + "/" + key))
      newdir = "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder\
/FileFinderData/AMI_cut/Arrays/Array1-01/" + id + "/" + key + "/" + id + ".file\
" + str(i) + ".wav"
      t1 = info[1]*1000 - 50
      t2 = info[2]*1000 + 50
      newAudio = AudioSegment.from_wav(dir)
      newAudio = newAudio[t1:t2]
      newAudio.export(newdir, format="wav")
      f.write("        " + newdir + ", \n")

def read_ami(filename, d={}):
  """ Returns dictionary representation of ami res file [filename] with [d]. """
  resfile = open(filename, 'r')
  key = None
  for line in resfile:
    if "[" in line : 
      key = line[1:-5]
      try: 
        d[key]
      except KeyError:
        d[key] = []
    elif (not "[" in line) and (not "]" in line):
      info = get_info(line)
      id = info[0]
      s1 = id.find('.')
      id = id[:s1]
      dir = "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/F\
ileFinderData/AMI/Arrays/Array1-01/" + id + "/audio/" + id + ".Array1-01.wav"
      print dir
      mfcc = get_mfcc(dir)
      d[key].append(mfcc)
  return d

def read_res(filename, d={}, maxlength=0):
  """ Returns dictionary representation of res file [filename] with [d]. """
  resfile = open(filename, 'r')
  key = None
  for line in resfile:
    if "[" in line :
      key = line[1:-5]
      try: 
        d[key]
      except KeyError:
        d[key] = []
    elif (not "[" in line) and (not "]" in line):
      dir = line.strip()[:-1]
      print dir
      mfcc = get_mfcc(dir)
      maxlength = max(maxlength, mfcc.shape[0])
      (d[key]).append(mfcc) #fuck this line
  return d,maxlength

def get_mfcc(wavfile):
  """ Returns the 13 MFCC values of [wavfile]. Can process .flac as well"""
  try: 
    (sig, rate) = sf.read(wavfile)
  except RuntimeError:
    i = wavfile.find("/wav/")
    wavfile = wavfile[:i] + "/flac/" + wavfile[i+5:-4] + ".flac"
    (sig, rate) = sf.read(wavfile)
  mfcc_feat = mfcc(sig,rate)
  d_mfcc_feat = delta(mfcc_feat, 2)
  fbank_feat = logfbank(sig,rate)
  return np.append(d_mfcc_feat, np.full((d_mfcc_feat.shape[0], 1), 0), axis=1)

def write_dict(d, directory):
  """ Writes dictionary of np arrays into a folder [d] """
  for cmd, arrlist in d.iteritems():
    np.savez(directory+cmd, arrlist)

def print_shapes(dir):
  """ Prints lengths of arrays stored in an .npz file
  Purely for testing/diagnostic purposes. """
  dic = np.load(dir)
  print dic.files
  v = dic["arr_0"]
  for elt in v: 
    print elt.shape

def dlen(dict):
  acc = 0
  for k, v in dict.iteritems():
    acc = acc + len(v)
    print k, len(v)
  # return acc

def merge_dic(d, cmdlist):
  new_dict = {}
  for k, v in d.iteritems():
    for cmd in cmdlist:
      if k.find(cmd) != -1:
        try:
          new_dict[cmd] = new_dict[cmd] + (v)
        except KeyError:
          new_dict[cmd] = v
  return new_dict

def pad_dict(d, maxlen):
  for k,v in d.iteritems():
    new_v = []
    for arr in v:
      shape = arr.shape
      pad = maxlen - shape[0]
      arr = np.pad(arr, ((0, pad),(0, 0)), 'constant', constant_values = (0))
      new_v.append(arr)
    d[k] = np.array(new_v)

# newsample = []
# for wav in sample:
#   shape = len(wav)
#   pad = 3 - shape
#   wav = np.pad(wav, ((0, pad),(0, 0)), 'constant', constant_values = (0))
#   newsample.append(wav)
# print np.array(newsample)


def main():
  m = 0
  dic = {}
  dic,m = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/libri_results.txt", dic, m)
  dic,m = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/surf_results.txt", dic, m)
  dic,m = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/vox_results.txt", dic, m)
  dic,m = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/vy_results.txt", dic, m)
  dic,m = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/wsj_results.txt", dic, m)
  dic,m = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/cut_ami_results.txt", dic, m)
  # print m
  pad_dict(dic, m)
  # dlen(dic)
  write_dict(dic, "./MFCCData/")
  # cut_ami("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/metadata.ami_results.txt")
  # merged_dic = merge_dic(dic, ["follow", "small", "medium", "large", "stop", "party"])
  # write_dict(merged_dic, "./MFCCData_merged/")
  # print_shapes("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/PSF/MFCCData/smaller.npz")

if __name__ == "__main__":
  main()