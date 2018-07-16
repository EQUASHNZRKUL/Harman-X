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
  List.filter (fun x -> x != ".DS_Store") (Array.to_list files)

(** [list_of_files' foldername] is the list representation of the contents of
  * directory [foldername] but with [foldername]/ concatenated to each one. *)
let list_of_files' foldername = 
  let files = Sys.readdir foldername in
  List.map (fun x -> foldername ^ "/" ^ x) (Array.to_list files)

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

type lineinfo = {id : string; starttime : float; endtime : float}

module AMID = struct 
  type t = lineinfo 
  let str i = "{id = " ^ i.id ^ "; starttime = " ^ (string_of_float i.starttime)
    ^ "; endtime = " ^ (string_of_float i.endtime) ^ "}" 
  let compare i1 i2 = 
    if i1.starttime < i2.starttime then `LT else if i1.starttime = i2.starttime 
    then `EQ else `GT
  let format fmt i = print_string (str i)
end

module S = MakeSetOfDictionary (StringD) (MakeTreeDictionary)
module D = MakeTreeDictionary (StringD) (S)

module Sami = MakeSetOfDictionary (AMID) (MakeTreeDictionary)
module Dami = MakeTreeDictionary (StringD) (Sami)

(** [print_list lst] prints the elements of string list [lst] *)
  let print_list lst = 
    let f x = print_string x in
    List.iter f lst

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

  (** [try_vox predes lst] is the file contents of predes/(first matching prompt
    * found in [lst]). Helper to [accesstext_vox], its necessary since Vox does
    * not have consistent directory structure. *)
  let rec try_vox predes lst = 
    match lst with 
    | [] -> raise (Sys_error "out of options in lst")
    | e::[] -> read_file (predes ^ e)
    | h::t -> 
      try
        read_file (predes ^ h)
      with Sys_error x -> 
        try_vox predes t

  (** [accesstext_vox folder data] is the file contents of a data point found 
    * at )folder/data/etc/(prompts-original)*)
  let accesstext_vox folder data = 
    let predes = folder ^ "/" ^ data ^ "/etc/" in
    let l = ["prompts-original"; "PROMPTS"; "prompt.txt"; "Transcriptions.txt";
            "prompts.txt"] in
    try_vox predes l 

  (** [accesswav_vox folder data wav] is the wav directory of a vox data point. 
    * ([folder]/[data]/wav/[wav].wav). *)
  let accesswav_vox folder data wav = 
    let predes = folder ^ "/" ^ data ^ "/wav/" in
    let wav' = try 
      let i = String.rindex wav '/' in
      String.sub wav i (String.length wav - i)
    with Not_found -> wav in
    predes ^ wav'^ ".wav"

  (** [accesstext_libri folder data] is the file contents of a libri data point
    * ([folder]/[data]/[data].trans.txt*)
  let accesstext_libri folder data = 
    let dest = folder ^ "/" ^ data ^ "/" ^ data ^ ".trans.txt" in
    read_file dest

  (** [accessflac_libri folder data wav] is the flac directory of a libri data
    * point ([folder]/[data]/[wav].flac) where [folder] is the libri directory, 
    * [data] is the directory of the data folder from [folder] and [wav] is the 
    * label. *)
  let accessflac_libri folder data wav = 
    (String.concat "/" [folder; data; wav]) ^ ".flac"

  (** [accesstext_surf folder data] is the file contents of a surf data point 
    * ([folder]/text/text.txt) *)
  let accesstext_surf folder data = 
    let dest = folder ^ "/text/text.txt" in
    read_file dest

  (** [accesswav_surf folder data wav] returns the wav directory of a surf data 
    * point ([folder]/[data]/wav). *)
  let accesswav_surf folder data wav = 
    String.concat "/" [folder; data; wav]

  (** [accesstext_vy folder data] returns the file contents of a vy data point 
    * transcription ([folder]/[data]/[data].wav.trn) *)
  let accesstext_vy folder data = 
    let dest = folder ^ "/" ^ data ^ "/" ^ data ^ ".wav.trn" in
    read_file dest

  (** [accesswav_vy folder data wav] returns the wav directory of a vy data 
    * point ([folder]/[data]/[data].wav) *)
  let accesswav_vy folder data wav = 
    String.concat "/" [folder; data; data] ^ ".wav"

(* Print Functions *)

  (** [print_value set] prints the elements of a set into channel [c]. *)
  let print_value c set = 
    let f k acc = 
      Printf.fprintf c "        %s, \n" k; acc in
    S.fold f [] set

  (** [print_value_ami c set] prints the elements of an AMI set into channel [c] *)
  let print_value_ami c set = 
    let f k acc = 
      Printf.fprintf c "        %s, \n" (AMID.str k); acc in
    Sami.fold f [] set

  (** [print_result dict] prints the assoc_list representation of [dict]. *)
  let print_result channel dict = 
    let f k v acc = 
      Printf.fprintf channel "\"%s\": [\n" k; 
      let acc = print_value channel v in
      Printf.fprintf channel "]\n";
      acc in
    D.fold f [] dict

  (** [print_result_ami channel ami] prints the assoc_list rep of [dict]. *)
  let print_result_ami channel ami = 
    let f k v acc = 
      Printf.fprintf channel "\"%s\": [\n" k; 
      let acc = print_value_ami channel v in
      Printf.fprintf channel "]\n";
      acc in
    Dami.fold f [] ami

(** [getCmdList str acc] is the list representation of the commands found in the
  * string [str], each separated by ';'. [acc] is the list so far. *)
let rec getCmdList str acc = 
  try 
    let j = String.index str ';' in
    let str' = (String.sub str (j+1) (String.length str - j-1)) in
    getCmdList (String.trim str') ((String.sub str 0 j)::acc)
  with Not_found-> (*print_string "NF - "; print_string str;*)
    str::acc

(** [flatten ds] is a script that "flattens" the LibriSpeech dataset [ds] into a
  * format better for [find_words]. It is meant to be executed multiple times
  * on separate LibriSpeech folders to combine them together into a single one.
  * The resulting LibriSpeechFlat folder stores the .flac and transcriptions
  * together in corresponding folders. *)
let flatten ds = 
  let outerlist = clean_list (list_of_files ds) [] in
  let f book = 
    let innerlist = clean_list (list_of_files (ds ^ "/" ^ book)) [] in
    let g user = 
      Sys.command ((
      "mv /Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/LibriSpeech_360/train-clean-360/" 
      ^ book ^ "/" ^ user) ^ 
      (" /Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/LibriSpeechFlat/" 
      ^ book ^ "-" ^ user)) in
    List.map g innerlist in
  List.map f outerlist

(** [unflatten ds] is a script that "unflattens" Vystidial data [ds] and groups 
  * the wav and transcription files together in their corresponding folders. *)
let unflatten ds = 
  let file_list = list_of_files ds in
  let f s = if s = ".DS_Store" then Sys.command "cd ." else
    let len = String.length s in
    let suff = String.sub s (len-4) 4 in
    let dest = if suff = ".trn" then String.sub s 0 (len-8) else String.sub s 0 (len-4) in
    (* let i = Sys.command ("mkdir /Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/Vystidial/" ^ dest) in *)
    let cmd = "mv " ^ ds ^ "/" ^ s ^ " /Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/Vystidial/" 
              ^ dest ^ "/" ^ s in
    Sys.command cmd in
  List.map f file_list

(** [ami_cleanup] is a script that moves files from array folders in [ds] and
  * flattens them into [target] but differs from [flatten ds] as it cleans up
  * the names so that they are referencible from *)
(* let ami_cleanup ds target = 
  let arr_list = list_of_files ds in
  let f arr = 
    let folder_list = list_of_files (ds ^ arr) in
    let g folder = 
      let dir =  *)

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

(** [contains s1 s2] is true if s2 exists in s1 as a substring (case-blind). 
  * (s1 contains s2) == (s1 <~= s2) *)
let (<~=) s1 s2 = 
  let s1 = String.lowercase_ascii s1 in
  let s2 = String.lowercase_ascii s2 in
  let re = Str.regexp_string s2 in
  try ignore (Str.search_forward re s1 0); true
  with Not_found -> false

(** [s -:- (i1, i2)] is the substring of s from i1 to i2 (inclusive) s[i1:i2] *)
let (-:-) s (i1,i2) = 
  String.sub s i1 (i2-i1+1)

(** [valid_lines] returns the valid wave file names from the [prompt_list] which
  * is the list of prompts, each element corresponding to a separate 5 sec wav
  * recording, that contain a word from [cmdlist]. If they do exist, they are
  * put into a dict as a value with the corresponding wavid as the key *)
let rec valid_lines prompt_list cmdlist audiofile dict prefix = 
  let f dict prompt = 
    (* print_endline prompt (*TEST: prints each file*) ; *)
    let g set c = if prompt <~= c then S.insert c set else set in
    let v = List.fold_left g S.empty cmdlist in
    if v != S.empty then
      let k = if prefix then 
      String.index prompt ' ' |> String.sub prompt 0 |> audiofile 
      else audiofile prompt in
      D.insert k v dict
    else dict in
  List.fold_left f dict prompt_list

(** [find_words cmdlist text audio foldername] is the (wav * prompt) list 
  * of data points in dataset [foldername] with prompt access_function of [text]
  * and wav location access_function of [audio] that contain an instance of any 
  * word found in [cmdlist]. *)
let find_words cmdlist text audio dataset prefix = 
  let f dict file = if file = ".DS_Store" then dict else
    let promptlist = text dataset file in
    valid_lines promptlist cmdlist (audio dataset file) dict prefix in 
  List.fold_left f D.empty (list_of_files dataset)

(** [get_info line] is the record extracting the information from an AMI line. *)
let get_info line = 
  let ids = String.index line '=' in
  let idf = String.index_from line (ids+2) '"' in
  let id' = String.sub line (ids+2) (idf-ids-2) in
  let starts = String.index_from line idf '=' in
  let startf = String.index_from line (starts+2) '"' in
  let start' = float_of_string (String.sub line (starts+2) (startf - starts - 2)) in
  let ends = String.index_from line startf '=' in
  let endf = String.index_from line (ends+2) '"' in
  let end' = float_of_string (String.sub line (ends+2) (endf - ends - 2)) in
  let vals = String.index line '>' in
  let valf = String.index_from line vals '<' in
  let val' = String.sub line (vals+1) (valf - vals - 1) in
  {id = id'; starttime = start'; endtime = end'}, val'

(** [ami_dict dir cmd_list] is the ami dictionary. Similar output to find_words
  * but with sets of records as values rather than sets of strings. *)
let ami_dict dir cmd_list = 
  let files = clean_list (list_of_files dir) [] in
  let clst line acc word = ((line <~= "/w>") && (line <~= word)) || acc in
  let f dictacc file = 
    let lines = read_file (dir ^ "/" ^ file) in
    let g dict line = 
      print_endline (line);
      let contains = List.fold_left (clst line) false cmd_list in
      if contains then 
        let info,k = get_info line in
        let valopt = Dami.find k dict in
        let set = (match valopt with | None -> Sami.empty | Some x -> x) in
        print_endline ("contained");
        Dami.insert k (Sami.insert info set) dict 
      else dict in
    List.fold_left g dictacc lines in
  List.fold_left f Dami.empty files

(** [surf_dict dir cmd_list] is the dictionary resulting from parsing the surf
  * database found at directory [dir] with commands from [cmd_list]. The result
  * dict has cmds as keys and sets of wav directories as values. *)
let surf_dict dir cmd_list = 
  let des = dir ^ "/text/text.txt" in
  let lines = read_file des in
  let line_f dict line = 
    let valcmds = List.filter (fun c -> line <~= c) cmd_list in
    if valcmds != [] then 
    let i = String.index line '.' in
    let wav = "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/SurfFolder/SurfData/" ^ (String.sub line 0 (i+4)) in
    let cmd_f dict cmd = 
      let valopt = D.find cmd dict in
      let s = (match valopt with 
      | None -> S.empty
      | Some v -> v) in
      let v' = S.insert wav s in
      print_endline cmd;
      D.insert cmd v' dict in
    List.fold_left cmd_f dict valcmds
    else dict in
  List.fold_left line_f D.empty lines

(** [wsj0_unbox dir] distributes the files found in wsj0 into a data folder,
  * ready to be processed by [find_words]. *)
let wsj0_unbox dir = (*wsj0*)
  let folder_list = clean_list (list_of_files dir) [] in
  let folder_f folder = (*sd_dt_05*)
    let data_list = clean_list (list_of_files (dir ^ "/" ^ folder)) [] in 
    let data_f data = (*00a*)
      let src = String.concat "/" [dir; folder; data] in
      let des = "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/WSJ0" in
      let des = String.concat "/" [des; folder^"_"^data] in
      let cmd = String.concat " " ["mv"; src; des] in
      Sys.command cmd in
    List.map data_f data_list in
  List.map folder_f folder_list

(** [txtify dir] converts the .dot files into the .txt files that are readable*)
let txtify dir = 
  let point_list = clean_list (list_of_files dir) [] in (*WSJ0*)
  let folder_f point = 
    let file_list = clean_list (list_of_files (dir ^ "/" ^ point)) [] in (*sx____*)
    let rename file = 
      let file = String.concat "/" [dir; point; file] in
      let l = String.length file in
      if (String.sub file (l-4) 4) = ".dot" then 
        (Sys.command ("mv " ^ file ^ " " ^ (String.sub file 0 (l-4)) ^ ".txt")) 
        else 0 in
    List.map rename file_list in
  List.map folder_f point_list

let wsj0_dict dir cmd_list= 
  let point_list = clean_list (list_of_files dir) [] in (*WSJ0*)
  let point_f dict point = 
    let file_list = clean_list (list_of_files (dir ^ "/" ^ point)) [] in (*sx____*)
    let txt_list = List.filter (fun x -> x <~= ".txt") file_list in
    let text_f pdict txt = 
      let prefix = String.sub txt 0 6 in
      let lines = read_file (String.concat "/" [dir; point; txt]) in
      let line_f dictacc prompt = 
        let cmd_f set cmd = if prompt <~= cmd then S.insert cmd set else set in
        let v = List.fold_left cmd_f S.empty cmd_list in
        if v != S.empty then
          let i = String.rindex prompt '(' in
          let l = String.length prompt in
          let loc = String.sub prompt (i+1) (l-i-2) in
          let wav = (String.concat "/" [dir; point; loc]) ^ "_1.wav" in
          D.insert wav v dictacc
        else dictacc in
      List.fold_left line_f pdict lines in
    List.fold_left text_f dict txt_list in
  List.fold_left point_f D.empty point_list

  let result_reader dir = 
    let result_txts = list_of_files dir in 
    let read_res accdict resfile = 
      let curr_key = ref "" in
      let line_list = read_file resfile in
      let dict_maker dict line = 
        if (line = "]") then dict else 
        if (line <~= ": [") then 
          let i = String.index line ':' in
          curr_key := (String.sub line 1 (i-2)); dict
        else 
          let v = String.trim line in 
          let set = D.find (!curr_key) dict in
          let val = (match set with
          | None -> S.insert v S.empty
          | Some s -> S.insert v s) in
          D.insert (!curr_key) val dict in
      List.fold_left dict_maker accdict line_list in
    List.fold_left read_res D.empty result_txts 

  let dir_accumulate dict des = 
    let res_dict = result_reader des in
    let dict_function k v dict = 
      let set_function e acc = 
        let i = String.rindex e '/' in
        let name = String.sub (i+1) (String.length - i - 3) in
        let des = "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/data/" ^ k ^ "/" in
        let cmd = String.concat " " ["cp";e;des] in
        acc + (Sys.command cmd) in
      S.fold set_function v in
    D.fold dict_function 0 res_dict 

          
        

let main () = 
  let simpleton = fun x y z -> x in
  let args = Sys.argv in
  let cmdlist = getCmdList argv.(1) [] in
  let dirpath = if argv.(2) = "" then "./FileFinderData" else argv.(2) in
  (* let taccess = accesstext_maker args.(3) args.(4) in *)
  let waccess = accesswav_maker args.(3) args.(5) in
  let res,txtout = (match argv.(6) with 
    | "vox" -> (find_words cmdlist accesstext_vox accesswav_vox dirpath true, 
                "vox_results.txt")
    | "libri" -> (find_words cmdlist accesstext_libri accessflac_libri dirpath 
                true, "libri_results.txt")
    | "surf" -> (surf_dict dirpath cmdlist, "surf_results.txt")
    | "vy" -> (find_words cmdlist accesstext_vy accesswav_vy dirpath false, 
              "vy_results.txt")
    | "ami" -> (D.empty, "ami_results.txt")
    | "wsj" -> print_endline "wsj"; (wsj0_dict dirpath cmdlist, "wsj_results.txt")
    | _ -> (D.empty,"")) in
  let cmd_dict = make_cmd_dict res D.empty in
  let oc = open_out ("results/" ^ txtout) in
  if argv.(6) = "ami" then 
    ignore(print_result_ami oc (ami_dict dirpath cmdlist))
  else if argv.(6) = "surf" then
    ignore (print_result oc res)
  else ignore (print_result oc cmd_dict);
  close_out oc;
  (* txtify "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/WSJ0" *)
  (* wsj0_unbox "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/WSJ0_meta/wsjdt/wsj0" *)
  (* flatten "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/LibriSpeech_360/train-clean-360" *)
  (* unflatten "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/Vystidial/data/" *)
  ;;

main ()