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
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi Filefinder 

exec:
# ./Filefinder cmdList(separated by ';') datasetdir datapointdir textdir wavdir
	ocamlopt -o Filefinder data.mli data.ml str.cmxa Filefinder.ml

eval:
# ./Filefinder cmdList(separated by ';') datasetdir datapointdir textdir wavdir
	ocamlopt -o Filefinder data.mli data.ml str.cmxa Filefinder.ml 
	./Filefinder "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/VoxForge" "" "etc/PROMPTS" "wav"
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi Filefinder

libri:
	ocamlopt -o main data.mli data.ml str.cmxa Filefinder.ml main.ml
	./main "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/LibriSpeechFlat" "" "" "" "libri"
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi main main.cmi main.cmx main.o

vox:
	ocamlopt -o Filefinder data.mli data.ml str.cmxa Filefinder.ml 
	./Filefinder "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/VoxForge" "" "etc/PROMPTS" "wav" "vox" 
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi Filefinder

surf: 
	ocamlopt -o Filefinder data.mli data.ml str.cmxa Filefinder.ml 
	./Filefinder "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/SurfFolder" "" "" "" "surf" 
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi Filefinder

vy:
	ocamlopt -o Filefinder data.mli data.ml str.cmxa Filefinder.ml 
	./Filefinder "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/Vystidial" "" "" "" "vy" 
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi Filefinder

ami:
	ocamlopt -o Filefinder data.mli data.ml str.cmxa Filefinder.ml 
	./Filefinder "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/AMI/ami_manual_1.6.1/words" "" "" "" "ami" 
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi Filefinder

wsj:
	ocamlopt -o Filefinder data.mli data.ml str.cmxa Filefinder.ml 
	./Filefinder "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/WSJ0" "" "" "" "wsj" 
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi Filefinder

all: 
	ocamlopt -o Filefinder data.mli data.ml str.cmxa Filefinder.ml 
	./Filefinder "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/LibriSpeechFlat" "" "" "" "libri"
	./Filefinder "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/VoxForge" "" "etc/prompts-original" "wav" "vox" 
	./Filefinder "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/SurfFolder" "" "" "" "surf" 
	./Filefinder "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/Vystidial" "" "" "" "vy" 
	./Filefinder "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/AMI/ami_manual_1.6.1/words" "" "" "" "ami" 
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi Filefinder

export: 
	ocamlopt -o Filefinder data.mli data.ml str.cmxa Filefinder.ml 
	./Filefinder "follow; follow me; stop; party; party mode; small; medium; large; small beam; medium beam; large beam" "export"
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi Filefinder
