from __future__ import division
import numpy
from python_speech_features import sigproc
from python_speech_features import base
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

from scipy.fftpack import dct
from scipy.stats import boxcox
import scipy.io.wavfile as wav

def dscc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
         nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, 
         ceplifter=22, appendEnergy=True, winfunc=lambda x:numpy.ones((x,))):
    feats, energies = base.fbank(signal, samplerate, winlen, winstep, nfilt, nfft, 
                            lowfreq, highfreq, preemph, winfunc)
    feats = base.delta(feats, 2) # OBTAIN DELTA
    feats = boxcox(feats)
    feats = numpy.log(feats)
    feats = dct(feats, type=2, axis=1, norm='ortho')[:,:numcep]
    feats = base.lifter(feats,ceplifter)
    if appendEnergy: feats[:,0] = numpy.log(energies) # replace first cepstral coefficient with log of frame energy
    feats = base.delta(feats, 2) #verify if 2 is right
    return feats

def main():
    (rate, sig) = wav.read ("en-0185.wav")
    mfcc_feat = mfcc(sig,rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate)
    print("fbank: ")
    print(fbank_feat[1:3,:])
    print("mfcc: ")
    print(d_mfcc_feat)
    dscc_feat = dscc(sig,rate)
    d_dscc_feat = delta(dscc_feat, 2)
    dbank_feat = logfbank(sig,rate)

if __name__ == "__main__":
    main()