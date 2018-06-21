open Str
open Sys

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
let accesstext_voxforge data = 
  let dest = String.concat "" [data; "/etc/prompts-original"] in
  read_file dest

(** [accesswav_voxforge user files] returns the list of wave destinations of the
  * files named [files] in the [user] folder for VoxForge data. *)
let accesswav_voxforge data files = 
  let f x = String.concat "" [data; "/wav/"; x; ".wav"] in
  List.map f files

(** [contains s1 s2] is true if s2 exists in s1 as a substring. 
  * (s1 contains s2) == (s1 <-= s2) *)
let (<-=) s1 s2 = 
  let re = Str.regexp_string s2 in
  try ignore (Str.search_forward re s1 0); true
  with Not_found -> false

(** [valid_lines] returns the valid wave file names from the [prompt_list] which
  * is the list of prompts, each element corresponding to a separate 5 sec wav
  * recording, that contain a word from [cmdlist]. If they do exist, they are
  * put into a list and returned in a tuple with their wav title. *)
let rec valid_lines prompt_list cmdlist = 
  let f acc prompt = 
    let filtered = List.filter (fun c -> prompt <-= c) cmdlist in
    if filtered != [] then (*command found in prompt, find wavid and cmd list*)
      let i = String.index prompt ' ' in
      (String.sub prompt 0 i, filtered) :: acc
    else acc in
  List.fold_left f [] prompt_list


  
(** [find_words' cmdlist text audio filelist acc] is the helper function to 
  * [find_words] and handles the recursive section. Returns a (wav * prompt)
  * list where the prompts contain an instance of any word from [cmdlist] and 
  * the wavs are audio representations of each prompt. The [text] and [audio] 
  * functions are dir -> prompt and dir -> wav location respectively. [acc] is 
  * the return list so far.*)
  (*TODO: Probably can be done with a fold instead*)
(* let rec find_words' cmdlist text audio filelist acc = 
  match filelist with
  | [] -> acc
  | h::t -> 
    let prompt = text h in
    
    let acc' = in 
    findwords' cmdlist accessfunc t acc' *)

(** [find_words cmdlist text audio foldername] is the (wav * prompt) list 
  * of data points in dataset [foldername] with prompt access_function of [text]
  * and wav location access_function of [audio] that contain an instance of any 
  * word found in [cmdlist]. *)
let find_words cmdlist text audio foldername = 
  let filelist = list_of_files foldername in
  let validList = valid_lines filelist cmdlist in

    (* let f acc file = 
      let prompt = text h in *)

  List.fold_left fold_func [] filelist 

  (* find_words' cmdlist text audio filelist [] *)