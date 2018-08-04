# JustinFolder
TensorFlow files & Data collection/processing code written during internship at Harman X

=================
Keyword Detection
=================

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

The top-level stuff in this folder is the engine of the code. FileFinder.ml and
Makefile are the files that will need to be edited if more data is added. 
FileFinderData holds all the datasets used for training. Filefinder.ml parses 
all the data in the folder and extracts instances of spoken commands. 

The Makefile is run as follows: 

make can be called with a keyword following it that affects which datasets it 
looks into. 
===========      ===========
Parameter         Description
===========      ===========
libri               processes the libri dataset
vox                 processes the voxforge dataset
surf                processes the surf dataset
vy                  processes the vystidial dataset
ami                 processes the AMI Corpus
wsj                 processes the WSJ0 dataset
all                 processes all the datasets above
export              copies all the results into a specified folder
exec                executes make all and make export as a single command
clean               removes all .o and other compile file garbage

# ./Filefinder cmdList(separated by ';') datasetdir datapointdir textdir wavdir
