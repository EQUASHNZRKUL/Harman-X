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
let accesstext_voxforge user = 
  let dest = String.concat "" [user; "/etc/prompts-original"] in
  read_file dest

(** [accesswav_voxforge user files] returns the list of wave destinations of the
  * files named [files] in the [user] folder. *)
let accesswav_voxforge user files = 
  let f x = String.concat "" [user; "/wav/"; x; ".wav"] in
  List.map f files

(** [valid_lines] returns the valid wave file names from the [prompt_list] which
  * is the list of prompts *)

let rec valid_lines 
  
(** [find_words' wordlist text audio filelist acc] is the helper function to 
  * [find_words] and handles the recursive section. Returns a (wav * prompt)
  * list where the prompts contain an instance of any word from [wordlist] and 
  * the wavs are audio representations of each prompt. The [text] and [audio] 
  * functions are dir -> prompt and dir -> wav location respectively. [acc] is 
  * the return list so far.*)
let rec find_words' wordlist text audio filelist acc = 
  match filelist with
  | [] -> acc
  | h::t -> 
    let prompt = text h in
    
    let acc' = in 
    findwords' wordlist accessfunc t acc'

(** [find_words wordlist text audio foldername] is the (wav * prompt) list 
  * of data points in dataset [foldername] with prompt access_function of [text]
  * and wav location access_function of [audio] that contain an instance of any 
  * word found in [wordlist]. *)
let find_words wordlist text audio foldername = 
  let filelist = list_of_files foldername in
    let f acc file = 
      let prompt = text h in

  List.fold_left fold_func [] filelist 

  (* find_words' wordlist text audio filelist [] *)