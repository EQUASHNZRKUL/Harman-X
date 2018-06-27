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

exec:
# ./Filefinder cmdList(separated by ';') datasetdir datapointdir textdir wavdir
	ocamlopt -o Filefinder data.mli data.ml str.cmxa Filefinder.ml
	./Filefinder "taught; average; all" "./FileFinderData" "" "etc/prompts-original" "wav"
	rm data.cmi data.o data.cmx Filefinder.cmx Filefinder.o Filefinder.cmi Filefinder
