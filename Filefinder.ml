open Str
open Sys
open Data
open Engine

(** [read_file filename] is the string list of the text found in [filename] each
  * elt being a different line.*)
let read_file filename = 
  let lines = ref [] in
  let chan = open_in filename in
  try 
    while true; do
      lines := input_line chan :: !lines
    done; !lines
  with End_of_file ->
    close_in chan;
    List.rev !lines

(** [list_of_files foldername] is the list representation of the contents of 
  * directory ./[foldername] *)
let list_of_files foldername = 
  let files = Sys.readdir foldername in
  Array.to_list files

(** [accesstext_voxforge folder] is the access function for VoxForge prompts. It
  * returns the text representation of the data found in location [folder]. *)
let accesstext_voxforge folder data = 
  let dest = String.concat "/" [folder; data; "etc/prompts-original"] in
  read_file dest

(** [accesswav_voxforge user files] returns the list of wave destinations of the
  * files named [files] in the [user] folder for VoxForge data. *)
let accesswav_voxforge folder data wav = 
  String.concat "" [folder; "/"; data; "/wav/"; wav; ".wav"]

(** [clean_list lst acc] is the list [lst] without the .DS_Store file in the
  * folder with [acc] as the accumulator. *)
  let rec clean_list lst acc = 
    match lst with 
    | [] -> acc
    | ".DS_Store"::t -> acc @ t
    | h::t -> clean_list t (List.rev (h::(List.rev acc)))


(*  --- Dictionary Section ---  *)

module StringD = struct
  type t = string
  let compare s1 s2 = 
  let cme = String.compare 
  (String.lowercase_ascii (s1)) (String.lowercase_ascii (s2)) in 
  if cme = 0 then `EQ else if cme > 0 then `GT else `LT  
  let format fmt d = print_string d
end

module S = MakeSetOfDictionary (StringD) (MakeTreeDictionary)
module D = MakeTreeDictionary (StringD) (S)

(** [contains s1 s2] is true if s2 exists in s1 as a substring (case-blind). 
  * (s1 contains s2) == (s1 <-= s2) *)
  let (<~=) s1 s2 = 
  let s1 = String.lowercase_ascii s1 in
  let s2 = String.lowercase_ascii s2 in
  let re = Str.regexp_string s2 in
  try ignore (Str.search_forward re s1 0); true
  with Not_found -> false

(** [valid_lines] returns the valid wave file names from the [prompt_list] which
  * is the list of prompts, each element corresponding to a separate 5 sec wav
  * recording, that contain a word from [cmdlist]. If they do exist, they are
  * put into a list and returned in a tuple with their wav title. *)
let rec valid_lines prompt_list cmdlist audiofile dict = 
  let f dict prompt = 
    let filtered = List.filter (fun c -> prompt <~= c) cmdlist in
    if filtered != [] then (*command found in prompt, find wavid and cmd list*)
      let k = String.index prompt ' ' |> String.sub prompt 0 |> audiofile in
      let v = 
      (audiofile (String.sub prompt 0 i), filtered) :: acc (*convert from assoc list to dict*)
    else dict in (**)
  List.fold_left f acc prompt_list (*processes prompt_list into dict[acc] (not list)*)

(** [find_words cmdlist text audio foldername] is the (wav * prompt) list 
  * of data points in dataset [foldername] with prompt access_function of [text]
  * and wav location access_function of [audio] that contain an instance of any 
  * word found in [cmdlist]. *)
let find_words cmdlist text audio dataset = 
  let filelist = clean_list (list_of_files dataset) [] in
    let f dict file = (*fold function to dict & process each file in dataset*)
      let promptlist = text dataset file in
      valid_lines promptlist cmdlist (audio file) dict in (*returns dict with all files so far*)
    List.fold_left f D.empty filelist (*should be dict of wavids -> StringD Set*)