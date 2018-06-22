exception Unimplemented
open Unix
open Str
open Data

module type Engine = sig
  type idx
  val index_of_dir : string -> idx
  val to_list : idx -> (string * string list) list
  val or_not  : idx -> string list -> string list -> string list
  val and_not : idx -> string list -> string list -> string list
  val format : Format.formatter -> idx -> unit
end

module MakeEngine
  (S:Data.Set with type Elt.t = string)
  (D:Data.Dictionary with type Key.t = string and type Value.t = S.t)
  : Engine
=
struct
  type idx = D.t

  (* [clean str] takes in a preword string and returns a word string, meaning it 
   * removes any characters that are not A-Z, a-z, or 0-9 from the string 
   * requires: [str] is a valid string representing a preword *)

  (* let clean str = 
    let cleaned = Str.global_replace (Str.regexp "[^A-z 0-9]+") " " str in
    let cleanList = Str.split (Str.regexp "[ \t]+") cleaned in
    cleanList *)

  let clean str = 
    try 
    let sf = Str.search_forward (Str.regexp "[A-z 0-9]+") str 0 in 
    let sb = Str.search_backward (Str.regexp "[A-z 0-9]+") str 
    (String.length str - 1) in 
    String.sub str sf ((sb-sf) + 1)
    with Not_found -> ""

  (* [getLineList linestr acc] is the list of words (strings) from [linestr]
   * requires: [linestr] is a valid string representing a line of text
   * [acc] is a valid list for accumulation *)
  let rec getLineList linestr acc = 
    let prewordList = Str.split (Str.regexp "[ \t]+") linestr in
    let prewordList = List.map String.trim prewordList in
    let wordList = List.map (clean) prewordList in
    let wordList2 = List.filter (fun x -> x <> "") wordList in 
    let wordList3 = List.map (fun s -> String.lowercase_ascii s) wordList2 in
    wordList3

  (* [setConcat lst s] is a set formed by combining all the elements of set [s] 
   * and list [lst] 
   * requires: [lst] is a valid 'a list
   * [s] is a valid set as defined in Data *)
  let rec setConcat lst s = 
    match lst with 
    | [] -> s
    | h::t -> setConcat t (S.insert h s)

  (* [getWordsLine acc file] is the set of words from a text file [file]
   * requires: [acc] is a valid accumulating Set as defined in Data 
   * [file] is a valid in_channel representing a file in the directory *)
  let rec getWordsLine acc file = 
    try 
    let lineStr = input_line file in
    let lineList = getLineList lineStr [] in
    getWordsLine (setConcat lineList acc) file
    with End_of_file -> acc

  (* [getWordsFile dir] is the set of words from all the text files in certain
   * directory [dir] 
   * requires: [dir] is a valid string that represents a file directory *)
  let getWordsFile dir = 
    open_in dir |> getWordsLine S.empty 

  (* [getWordsDict filename dict] is a dictionary updated by inserting words 
   * from the directory [filename] with the values being the set of files in 
   * which those words are found and they keys being the individual words
   * requires: [filename] is a valid string (set element) representing a 
   * directory 
   * [dict] is a valid dictionary that will be updated through insertion *)
  let getWordsDict filename dict =
    let wordSet = getWordsFile filename in
    let updateDict word idxDict =
      let fileSet = D.find word idxDict in
      let newfSet = match fileSet with
        | None -> S.insert filename S.empty
        | Some words -> S.insert filename words in 
      D.insert word newfSet idxDict in 
    S.fold updateDict dict wordSet


  (* [extr dir d idxDict] returns a dictionary of files in a directory [dir]. 
   * The keys of the dictionary are individual words within the text files and 
   * values are sets (as defined in Data) of files that contain those words  
   * requires: [dir] is a valid Unix.dir_handle (directory)
   * [d] is a valid string representing a directory 
   * [idxDict] is a valid accumulation dictionary *)
  let rec extr dir d idxDict = try 
    let file = Unix.readdir dir in
    let filename = d ^ "/" ^ file in
    let len = String.length filename in
    if (String.sub filename (len-4) 4) = ".txt" then 
      let wordDict = getWordsDict filename idxDict in
      extr dir d wordDict (*insert .txt files only*)
    else 
    extr dir d idxDict
    with End_of_file -> idxDict

  (* [case_blind_mem k lst] is the boolean which works the same way as List.mem
   * except with the condition that this function ignores case 
   * requires: [k] is a valid string 
   * [lst] is a valid string list *)
   let rec case_blind_mem k lst = 
    match lst with 
    | [] -> false
    | h::t -> if String.lowercase_ascii k = String.lowercase_ascii h then true 
              else case_blind_mem k t

  (* [case_blind_mem k lst] is the boolean which works the same way as List.mem
   * except with the conditions that this function ignores case and operates 
   * with dictionary keys instead of elements of a list
   * requires: [k] is a valid string 
   * [d] is a valid dictionary with strings as keys *)
  let case_blind_mem_d key d =
    let f k v acc = if (String.lowercase_ascii k) = (String.lowercase_ascii key)
    then true else acc || false in
    D.fold f false d


  (* [clean_idx idx] is an idx with combined values for case_blind keys. This
   * means that if a word appears twice because of differing cases, this fxn 
   * will mix them into the lowercase version and union the file sets that 
   * were the values associated with the files. 
   * requires: [idx] is a valid index of type D.t *)
  let clean_idx idx = 
    let f k v acc = if case_blind_mem_d k acc then
      let oldVal = 
        match (D.find (String.lowercase_ascii k) acc) with Some v -> v | None -> v in 
        (*should never match with None since vs come from the dict. *)
      let newVal = S.union oldVal v in
      D.insert (String.lowercase_ascii k) newVal acc
    else D.insert (String.lowercase_ascii k) v acc in 
    D.fold f D.empty idx

  let index_of_dir d =
    try 
    let dir = Unix.opendir d in
    extr dir d D.empty 
    with Unix.Unix_error (Unix.ENOENT, _, _)
    -> raise Not_found

  let to_list idx =
    let to_elt k v acc = (k, S.to_list v) :: acc in (*convert to strings*)
    D.fold to_elt [] idx

  (* [dirSet idx accSet] is a set of all the files contained in all the values 
   * of the dictionary [idx]
   * requires: [idx] is a valid index of type D.t 
   * [accSet] is a valid S.t (Set) ready to be accumulated *) 
  let dirSet idx accSet= 
    let f k v acc = S.union acc v in D.fold f S.empty idx

  let or_not idx ors nots =
    let f k v acc = 
      if (case_blind_mem k ors) then S.union v acc else acc in 
    let g k v acc = 
      if (case_blind_mem k nots) then S.difference acc v else acc in
    let orSet = D.fold f S.empty idx in
    let norSet = D.fold g orSet idx in
    S.to_list norSet

  (* [and_in_idx idx a] is true if [a] is a key in [idx]. False otherwise.
   * requires: idx is valid index of type D.t 
   * [a] is a valid string representing a key of an index *)
  let and_in_idx idx a = 
    let f k v acc = if (String.lowercase_ascii k) = (String.lowercase_ascii a)
    then true else acc in
    D.fold f false idx 

  (* [all_ands idx ands] is true if all strings in [ands] are keys of [idx]. 
   * Functions the same way as the function above, but for multiple keys this 
   * time, as opposed to one single string. 
   * requires: idx is valid index of type D.t 
   * [and] is a valid string list representing a list of keys of an index *)
  let rec all_ands idx ands = 
    match ands with 
    | [] -> true
    | h::t -> if and_in_idx idx h then all_ands idx t else false

  let and_not idx ands nots =
    if (all_ands idx ands) then 
    let f k v acc = 
      if (case_blind_mem k ands) then S.intersect v acc else acc in 
    let g k v acc = 
      if (case_blind_mem k nots) then S.difference acc v else acc in
    let andSet = D.fold f (dirSet idx S.empty) (idx) in
    let norSet = D.fold g andSet idx in
    S.to_list norSet
    else []

  let format fmt idx =
    D.format fmt idx (* TODO: improve if you wish *)
end

module TrivialEngine =
struct
  type idx = unit
  let index_of_dir d = ()
  let to_list idx = []
  let or_not idx ors nots = []
  let and_not idx ands nots = []
  let format fmt idx = ()
end

module StringD = struct
  type t = string
  let s = " "
  let compare s1 s2 = if String.compare s1 s2 = 0 then `EQ else 
  if String.compare s1 s2 > 0 then `GT else `LT
  let format fmt d = print_string d
end

module Set = MakeSetOfDictionary (StringD) (MakeListDictionary)
module Dict = MakeListDictionary (StringD) (Set)

module Set2 = MakeSetOfDictionary (StringD) (MakeTreeDictionary)
module Dict2 = MakeTreeDictionary (StringD) (Set2)

module ListEngine = MakeEngine (Set) (Dict)
(* TODO: replace [TrivialEngine] in the line above with
   an application of [MakeEngine] to some appropriate parameters. *)

module TreeEngine = MakeEngine (Set2)(Dict2)
(* TODO: replace [TrivialEngine] in the line above with
   an application of [MakeEngine] to some appropriate parameters. *)
