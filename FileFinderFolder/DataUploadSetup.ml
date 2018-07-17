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

(** [result_reader dir] is the dict representation of the result textfiles @ 
  * [dir]. *)
let result_reader dir = 
  let result_txts = list_of_files dir in 
  let read_res accdict resfile = 
    let curr_key = ref "" in
    let line_list = read_file resfile in
    let dict_maker dict line = 
      if (line <~= "]") then dict else
      if (line <~= ": [") then 
        let i = String.index line ':' in
        print_endline ("in section 2: " ^ line);
        curr_key := (String.sub line 1 (i-2)); dict
      else 
        let v = String.trim line in 
        print_endline ("in section 3: " ^ line);
        let set_opt = D.find (!curr_key) dict in
        let set = (match set_opt with
        | None -> S.empty
        | Some x -> x) in
        D.insert (!curr_key) (S.insert v set) dict in
    List.fold_left dict_maker accdict line_list in
  List.fold_left read_res D.empty result_txts 
    
(** [dir_accumulate] copies the wav files found in [dict] to [des] in 
  * folders corresponding to keys. *)
let dir_accumulate res_dict predes = 
  let dict_function k v acc = 
    let set_function e acc = 
      let i = String.rindex e '/' in
      let name = String.sub e (i+1) ((String.length e) - i - 3) in
      let trash = Sys.command ("mkdir " ^ predes ^ k) in
      let des = predes ^ k ^ "/" ^ name in
      let cmd = String.concat " " ["cp";e;des] in
      acc + (Sys.command cmd) in
    S.fold set_function 0 v in
  D.fold dict_function 0 res_dict 

let main () = 
  print_endline "asdf";
  print_endline "wtf where are you"
  let results = result_reader "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/" in
  dir_accumulate results "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/data/"