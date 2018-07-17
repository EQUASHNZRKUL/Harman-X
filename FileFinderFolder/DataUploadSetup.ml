open Filefinder
open Data
open Sys

module StringD = struct
  type t = string
  let str s = s ^ ""
  let compare s1 s2 = 
    let cme = String.compare 
    (String.lowercase_ascii (s1)) (String.lowercase_ascii (s2)) in 
    if cme = 0 then `EQ else if cme > 0 then `GT else `LT  
  let format fmt d = print_string d
end

module S = MakeSetOfDictionary (StringD) (MakeTreeDictionary)
module D = MakeTreeDictionary (StringD) (S)



let main () = 
  print_endline "asdf";
  print_endline "wtf where are you"
  let results = result_reader "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/" in
  dir_accumulate results "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/data/"