from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

import scipy.io.wavfile as wav
import soundfile as sf
import numpy as np

def get_info(infostr):
    """ Returns tuple of metadata from ami_metadata file. [infostr] is one line
    of an ami_metadata file. Tuple is in format of cmd * starttime * endtime."""
    i1 = infostr.find("id = ")
    i2 = infostr.find("starttime = ")
    i3 = infostr.find("endtime = ")
    id = infostr[i1+5:i2-2]
    st = float(infostr[i2+12:i3-2])
    et = float(infostr[i3+10:-3])
    return (id, st, et)

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
            info = get_info(line)
            id = info[0]
            s1 = id.find('.')
            id = id[:s1]
            dir = "/Users/justinkae/Documents/TensorflowPractice/FinderFolderFolder/FinderFolderData/AMI/data/" + id + "/audio/" + id 
            print dir
            mfcc = get_mfcc(dir)
            d[key].append(mfcc)
    return d

def read_res(filename, d={}):
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
            (d[key]).append(mfcc) #fuck this line
    return d

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
    return d_mfcc_feat

def write_dict(d):
    """ Writes dictionary of np arrays into a folder [d] """
    for cmd, arrlist in d.iteritems():
        np.savez("./MFCCData/"+cmd, arrlist)

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

def main():
    dic = {}

    read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/libri_results.txt", dic)
    print ("**libri: ")
    dlen(dic)

    read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/surf_results.txt", dic)
    print ("**surf: ")
    dlen(dic)

    read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/vy_results.txt", dic)
    print ("**vy: ")
    dlen(dic)

    read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/ami_results.txt", dic)
    print ("**wsj: ")
    dlen(dic)

    # dic = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/vox_results.txt", dic)
    # print dlen(dic)    
    # dic = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/vy_results.txt", dic)
    # print dlen(dic)
    # dic = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/wsj_results.txt", dic)
    # print dlen(dic)
    # dic = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/ami_results.txt", dic)
    # print dlen(dic)

    # dic = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/libri_results.txt", dic)
    # dic = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/surf_results.txt", dic)
    # dic = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/vox_results.txt", dic)
    # dic = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/vy_results.txt", dic)
    # dic = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/wsj_results.txt", dic)
    # dic = read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/ami_results.txt", dic)

    # write_dict(dic)
    # print_shapes("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/PSF/MFCCData/smaller.npz")

if __name__ == "__main__":
    main()