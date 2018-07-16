open Filefinder
open Sys

let result_reader dir = 
  let result_txts = list_of_files dir in 
  let read_res resfile accdict = 
    let curr_key = ref "" in
    let line_list = read_file resfile in
    

let main () = 
  let results = result_reader "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/" in
