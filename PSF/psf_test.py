from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

import scipy.io.wavfile as wav

def read_res(dir):
    resfile = open(dir, 'r')
    key = None
    d = {}
    for line in resfile:
        if "[" in line :
            key = line[1:-4]
        if (not "[" in line) or (not "]" in line):
            dir = line.strip()[:-1]
            mfcc = get_mfcc(dir)
            d[key] = d[key].append(mfcc)

def get_mfcc(wavfile):
    (rate, sig) = wav.read (wavfile)
    mfcc_feat = mfcc(sig,rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate)
    return d_mfcc_feat

def main():


if __name__ == "__main__":
    main()