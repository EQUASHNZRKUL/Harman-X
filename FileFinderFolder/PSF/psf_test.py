from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

from numpy import savez 

import scipy.io.wavfile as wav
import soundfile as sf

def get_info(infostr):
    i1 = infostr.find("id = ")
    i2 = infostr.find("starttime = ")
    i3 = infostr.find("endtime = ")
    id = infostr[i1+5:i2-2]
    st = float(infostr[i2+12:i3-2])
    et = float(infostr[i3+10:-3])
    return (id, st, et)

def read_ami(filename, d={}):
    resfile = open(filename, 'r')
    key = None
    for line in resfile:
        if "[" in line : 
            key = line[1:-5]
            d[key] = []
        if (not "[" in line) and (not "]" in line):
            info = get_info(line)
            id = info[0]
            s1 = id.find('.')
            id = id[:s1]
            dir = "/Users/justinkae/Documents/TensorflowPractice/FinderFolderFolder/FinderFolderData/AMI/data/" + id + "/audio/" + id 
            print dir
            mfcc = get_mfcc(dir)
            d[key] = d[key] + [mfcc]
    return d

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
    for cmd, arrlist in d.iteritems():
        savez("./MFCCData/"+cmd, arrlist)

def main():
    dic = {}
    read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/libri_results.txt", dic)
    read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/surf_results.txt", dic)
    read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/vox_results.txt", dic)
    read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/vy_results.txt", dic)
    read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/ami_results.txt", dic)
    read_res("/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/wsj_results.txt", dic)
    write_dict(dic)

if __name__ == "__main__":
    main()