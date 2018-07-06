open Str
open Sys
open Data

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
  let str s = s ^ ""
  let compare s1 s2 = 
    let cme = String.compare 
    (String.lowercase_ascii (s1)) (String.lowercase_ascii (s2)) in 
    if cme = 0 then `EQ else if cme > 0 then `GT else `LT  
  let format fmt d = print_string d
end

module S = MakeSetOfDictionary (StringD) (MakeTreeDictionary)
module D = MakeTreeDictionary (StringD) (S)

(** [print_list lst] prints the elements of string list [lst] *)
  let print_list lst = 
    let f x = print_string x in
    List.iter f lst

(** [contains s1 s2] is true if s2 exists in s1 as a substring (case-blind). 
  * (s1 contains s2) == (s1 <~= s2) *)
  let (<~=) s1 s2 = 
  let s1 = String.lowercase_ascii s1 in
  let s2 = String.lowercase_ascii s2 in
  let re = Str.regexp_string s2 in
  try ignore (Str.search_forward re s1 0); true
  with Not_found -> false

(** [valid_lines] returns the valid wave file names from the [prompt_list] which
  * is the list of prompts, each element corresponding to a separate 5 sec wav
  * recording, that contain a word from [cmdlist]. If they do exist, they are
  * put into a dict as a value with the corresponding wavid as the key *)
let rec valid_lines prompt_list cmdlist audiofile dict = 
  let f dict prompt = 
    let g set c = if prompt <~= c then S.insert c set else set in
    let v = List.fold_left g S.empty cmdlist in
    if v != S.empty then
      let k = String.index prompt ' ' |> String.sub prompt 0 |> audiofile in
      D.insert k v dict
    else dict in
  List.fold_left f dict prompt_list

(** [find_words cmdlist text audio foldername] is the (wav * prompt) list 
  * of data points in dataset [foldername] with prompt access_function of [text]
  * and wav location access_function of [audio] that contain an instance of any 
  * word found in [cmdlist]. *)
let find_words cmdlist text audio dataset = 
  let filelist = clean_list (list_of_files dataset) [] in
    let f dict file = 
      let promptlist = text dataset file in
      valid_lines promptlist cmdlist (audio dataset file) dict in 
    List.fold_left f D.empty filelist

(** ACCESS FUNCTIONS **)

  (** [accesstext_maker datadir textdir] is an access text function maker. Given 
    * the path from the dataset folder to the data folder [datdir] and the path 
    * from the data folder to the text transcript file [textdir] such that they 
    * fulfill foldername/[datdir]/data/[textdir]. The resulting function returns
    * the text location for a given [folder] and [data].*)
    let accesstext_maker datadir textdir = fun folder data -> 
    let dest = String.concat "/" [folder; datadir; data; textdir] in
    read_file dest

  (** [accesswav_maker datadir wavdir] is an access wav function maker. Given
    * the path from the dataset folder to the data folder [datdir] and the path
    * from the data folder to the wav audio file [wavdir] such that they 
    * fulfill foldername/[datdir]/data/[wavdir]/wav.wav. The resulting function returns
    * the text location for a given [folder], [data], [wav].*)
  let accesswav_maker datadir wavdir = fun folder data wav -> 
    let des = String.concat "/" [folder; datadir; data; wavdir; wav] in
    String.concat "" [des; ".wav"]

  let accesstext_vox folder data = 
    let predes = folder ^ "/" ^ data in
    let des = predes ^ "/etc/prompts-original" in
    try
      read_file des
    with Sys_error x -> 
      try 
        let des = predes ^ "/etc/PROMPTS" in
        read_file des
      with Sys_error x ->
        let des = predes ^ "/etc/Transcriptions.txt" in
        read_file des


  let accesstext_libri folder data = 
    let dest = folder ^ "/" ^ data ^ "/" ^ data ^ ".trans.txt" in
    read_file dest

  let accessflac_libri folder data wav = (String.concat "/" [folder; data; wav]) ^ ".flac"

  let accesstext_surf folder data = 
    let dest = folder ^ "/text/text.txt" in
    read_file dest

  let accesstext_vy folder data = 
    let dest = folder ^ "/" ^ data ^ "/" ^ data ^ ".wav.trn" in
    read_file dest

(* Print Functions *)

  (** [print_value set] prints the elements of a set. *)
  let print_value c set = 
    let f k acc = 
      Printf.fprintf c "        %s, \n" k; acc in
    S.fold f [] set

  (** [print_result dict] prints the assoc_list representation of [dict]. *)
  let print_result channel dict = 
    let f k v acc = 
      Printf.fprintf channel "\"%s\": [\n" k; 
      let acc = print_value channel v in
      Printf.fprintf channel "]\n";
      acc in
    D.fold f [] dict

(** [getCmdList str acc] is the list representation of the commands found in the
  * string [str], each separated by ';'. [acc] is the list so far. *)
let rec getCmdList str acc = 
  try 
    let j = String.index str ';' in
    let str' = (String.sub str (j+1) (String.length str - j-1)) in
    getCmdList (String.trim str') ((String.sub str 0 j)::acc)
  with Not_found-> (*print_string "NF - "; print_string str;*)
    str::acc

(* flattens the directory *)
let flatten ds = 
  let outerlist = clean_list (list_of_files ds) [] in
  let f book = 
    let innerlist = clean_list (list_of_files (ds ^ "/" ^ book)) [] in
    let g user = 
      Sys.command ((
      "mv /Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/LibriSpeech_500/train-other-500/" 
      ^ book ^ "/" ^ user) ^ 
      (" /Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/LibriSpeechFlat/" 
      ^ book ^ "-" ^ user)) in
    List.map g innerlist in
  List.map f outerlist

let unflatten ds = 
  let file_list = clean_list (list_of_files ds) [] in
  let f s = 
    let len = String.length s in
    let suff = String.sub s (len-4) 4 in
    let dest = if suff = ".trn" then String.sub s 0 (len-8) else String.sub s 0 (len-4) in
    let i = Sys.command ("mkdir /Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/vystidial/" ^ dest) in
    let cmd = "mv " ^ ds ^ "/" ^ s ^ " /Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/vystidial/" 
              ^ dest ^ "/" ^ s in
    Sys.command cmd in
  List.map f file_list

(** [make_cmd_dict word_dict cmd_dict] is the command dictionary (keys: commands
  * values: sets of wavids) made from the wav dictionary [word_dict] (keys: wavs
  * values: sets of commands) where cmd_dict acts like an accumulator. *)
let rec make_cmd_dict word_dict cmd_dict = 
  let word_opt = D.choose word_dict in
  match word_opt with 
  | None -> cmd_dict
  | Some (wav, cmd_set) -> 
    let f cmd acc_dict = (*insert the cmd into the acc_dict list with wav as key *)
      let val_opt = D.find cmd acc_dict in
      let v' = (match val_opt with
      | None -> (S.insert wav S.empty)
      | Some wav_set -> (S.insert wav wav_set)) in
        D.insert cmd v' acc_dict in
    let cmd_dict' = S.fold f cmd_dict cmd_set in
    let word_dict' = D.remove wav word_dict in
    make_cmd_dict word_dict' cmd_dict'
    

let main () = 
  let simpleton = fun x y z -> x in
  let args = Sys.argv in
  let cmdlist = getCmdList argv.(1) [] in
  let dirpath = if argv.(2) = "" then "./FileFinderData" else argv.(2) in
  (* let taccess = accesstext_maker args.(3) args.(4) in *)
  let waccess = accesswav_maker args.(3) args.(5) in
  let res,txtout = (match argv.(6) with 
  | "vox" -> (
    find_words cmdlist accesstext_vox waccess dirpath, "vox_results.txt")
  | "libri" -> (find_words cmdlist accesstext_libri accessflac_libri dirpath, "libri_results.txt")
  | "surf" -> (find_words cmdlist accesstext_surf simpleton dirpath, "surf_results.txt")
  | "vy" -> (find_words cmdlist accesstext_vy simpleton dirpath, "vy_results.txt")
  | _ -> D.empty,"") in
  let cmd_dict = make_cmd_dict res D.empty in
  let oc = open_out ("results/" ^ txtout) in
  print_result oc cmd_dict;
  close_out oc;
  (* flatten "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/LibriSpeech_500/train-other-500" *)
  (* unflatten "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/vystidial/train/" *)
  ;;

main ()