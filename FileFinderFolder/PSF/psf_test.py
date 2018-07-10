from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

from numpy import savez 

import scipy.io.wavfile as wav

def read_res(filename, d={}):
    resfile = open(filename, 'r')
    key = None
    for line in resfile:
        if "[" in line :
            key = line[1:-5]
            d[key] = []
        if (not "[" in line) and (not "]" in line):
            dir = line.strip()[:-1]
            print dir
            mfcc = get_mfcc(dir)
            d[key] = d[key] + [mfcc]
    return d

def get_mfcc(wavfile):
    (rate, sig) = wav.read (wavfile)
    mfcc_feat = mfcc(sig,rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate)
    return d_mfcc_feat

def write_dict(d):
    for cmd, arrlist in d.iteritems():
        savez("./MFCCData/"+cmd, arrlist)

def main():
    dic = {}
    read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/surf_results.txt", dic)
    read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/vox_results.txt", dic)
    read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/vy_results.txt", dic)
    write_dict(dic)

if __name__ == "__main__":
    main()