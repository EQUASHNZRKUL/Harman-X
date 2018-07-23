# test:
# 	ocamlbuild -use-ocamlfind test_main.byte && ./test_main.byte

compile:
	ocamlbuild -use-ocamlfind data.cmo data.cmi Filefinder.cmo

check:
	bash checkenv.sh && bash checktypes.sh

# zip:
# 	zip a3src.zip *.ml*
	
# zipcheck:
# 	bash checkzip.sh

clean:
	ocamlbuild -clean
	rm -f a3src.zip
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi main main.cmi main.cmx main.o

exec:
# ./Filefinder cmdList(separated by ';') datasetdir datapointdir textdir wavdir
	ocamlopt -o main data.mli data.ml str.cmxa Filefinder.ml main.ml

eval:
# ./Filefinder cmdList(separated by ';') datasetdir datapointdir textdir wavdir
	ocamlopt -o main data.mli data.ml str.cmxa Filefinder.ml main.ml
	./main "vox" "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/VoxForge" "" "etc/PROMPTS" "wav"
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi main main.cmi main.cmx main.o

libri:
	ocamlopt -o main data.mli data.ml str.cmxa Filefinder.ml main.ml
	./main "libri" "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/LibriSpeechFlat" "" "" ""
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi main main.cmi main.cmx main.o

vox:
	ocamlopt -o main data.mli data.ml str.cmxa Filefinder.ml main.ml
	./main "vox" "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/VoxForge" "" "etc/PROMPTS" "wav" 
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi main main.cmi main.cmx main.o

surf: 
	ocamlopt -o main data.mli data.ml str.cmxa Filefinder.ml main.ml
	./main "surf" "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/SurfFolder" "" "" ""
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi main main.cmi main.cmx main.o

vy:
	ocamlopt -o main data.mli data.ml str.cmxa Filefinder.ml main.ml
	./main "vy" "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/Vystidial" "" "" "" 
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi main main.cmi main.cmx main.o

ami:
	ocamlopt -o main data.mli data.ml str.cmxa Filefinder.ml main.ml
	./main "ami" "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/AMI/ami_manual_1.6.1/words" "" "" "" 
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi main main.cmi main.cmx main.o

wsj:
	ocamlopt -o main data.mli data.ml str.cmxa Filefinder.ml main.ml
	./main "wsj" "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/WSJ0" "" "" "" 
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi main main.cmi main.cmx main.o

all: 
	ocamlopt -o main data.mli data.ml str.cmxa Filefinder.ml main.ml
	./main "libri" "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/LibriSpeechFlat" "" "" "" 
	./main "vox" "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/VoxForge" "" "etc/prompts-original" "wav" 
	./main "surf" "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/SurfFolder" "" "" "" 
	./main "vy" "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/Vystidial" "" "" "" 
	./main "ami" "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/AMI/ami_manual_1.6.1/words" "" "" "" 
	./main "wsj" "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/WSJ0" "" "" "" 
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi main main.cmi main.cmx main.o

export: 
	ocamlopt -o main data.mli data.ml str.cmxa Filefinder.ml main.ml
	./main "export" "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam"
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi main main.cmi main.cmx main.o
