# Keyword Detection
TensorFlow files & Data collection/processing code written during internship at Harman X

Overview:
=========

This project is split into 2 distinct folders. FileFinderFolder contains the code
used to gather data and VGG contains the tensorflow files used to create a CNN, 
train it, and evaluate it. It's technically a misnomer and should be something
like 'CNN' or 'Machine_Learning' instead, but I'm too scared to change its name
in fear of breaking something, and its the last day of my internship at 5:05 PM
as I write this, so I don't really have time to fix that right now. I really 
shouldn't have pushed this off so much. 

FileFinderFolder:
=================

Call **make all** in order to extract command files and store the MFCCData into FileFinderFolder/PSF/MFCC_Data_folder. This directory can be changed in [dir_to_data.py]. 

The top-level stuff in this folder is the engine of the code. FileFinder.ml and
Makefile are the files that will need to be edited if more data is added. 
FileFinderData holds all the datasets used for training. Filefinder.ml parses 
all the data in the folder and extracts instances of spoken commands. 

The Makefile is run as follows: 

**make** can be called with a command following it that affects which datasets it 
looks into. 

| Command | Description |
| --- | --- |
| libri | processes the libri dataset |
| vox | processes the voxforge dataset |
| surf | processes the surf dataset |
| vy | processes the vystidial dataset |
| ami | processes the AMI Corpus |
| wsj | processes the WSJ0 dataset |
| all | processes all the datasets above |
| export | copies all the results into a specified folder & runs MFCC processing on them |
| exec | executes make all and make export as a single command |
| clean | removes all .o and other compile file garbage |

The Data parsing commands must be followed by the following arguments to work. 
If a new dataset format is introduced, it must be implemented in here as well in
[exec] and [all] at the very least. 

This is the full command: 
./main cmdList(separated by ';') datasetdir datapointdir textdir wavdir

| Index | Argument | Explanation |
| --- | --- | --- |
| 0 | ./main | Executable |
| 1 | command list | list of commands you're looking for in FileFinderData (separated by ;)
| 2 | datasetfolder | directory of dataset | 
| 3 | datapointdir | directory of a datapoint from dataset directory (only needed for non-defaults |
| 4 | textdir | directory of a textfile from a datapointdir |
| 5 | wavdir | directory of a wavfile from a datapointdir |

Any dataset that cannot refer to its sound files with "datasetfolder/[datapointdir]/data/[wavdir]/wav.wav" where foldername, data, wav are specific properties of a given datapoint, and [datapointdir] and [wavdir] are constant throughout a dataset, needs a hard-coded accesswav function. Refer to [accesswav_vox] and [accesswav_ami]. It is a simple function that simply takes a datasetfolder, datapoint, and wavname and returns the directory. 

VGG
===

Call **python VGG_all.py** to build, train, and evaluate the neural network. You can make your own CNN by adjusting lines 55 & 176 in VGG_all.py and making your own build function. Make sure they call the same build function though; you must eval and train on the same graph. 

VGG.py contains the guts of the Neural Network. VGG_all.py is used to train and evaluate. It takes a large number of arguments. 

| Flag | Default Value | Description |
| --- | --- | --- |
| train_dir | checkpoints/ | Directory where to write graph and checkpoint files |
| log_frequency | 50 | How often to log results to console|
| max_steps | 1000 | Number of batches to run |
| log_device_placement | False | Whether to log device placement |
| eval_dir | ./eval_logs | Directory where to write event logs |
| eval_data | ../FileFinderFolder/PSF/MFCCData_folder/MFCCData_split/test.npz | Testing data |
| checkpoint_dir | ./checkpoints | Directory where to read model checkpoints |
| eval_interval_secs | 300 | How often to run the eval |
| num_examples | 1000 | Number of examples to run |
| run_once | False | Whether to run only once |

I will be updating this README later. You can find it on the github page. 