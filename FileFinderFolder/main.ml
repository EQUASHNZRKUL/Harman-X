open Filefinder
open Str
open Sys
open Data

let main () = 
  let simpleton = fun x y z -> x in
  let args = Sys.argv in
  let cmdlist = getCmdList argv.(2) [] in
  let dirpath = if argv.(3) = "" then "./FileFinderData" else argv.(3) in
  match args.(1) with
  | "export" ->     
    let results = total_res_dict "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/" in
    let trash = dir_accumulate results "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/data/" in
    let trash = dir_accumulate_merged results "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/data_merged/" cmdlist in
    ignore(trash)
  | "ami" -> 
    let oc = open_out ("results/metadata.ami_results.txt") in
    ignore(print_result_ami oc (ami_dict dirpath cmdlist));
    close_out oc;
    let oc = open_out ("results/ami_results.txt") in
    ami_textify "results/metadata.ami_results.txt" oc
  | "surf" -> 
    let res, txtout = (surf_dict dirpath cmdlist, "surf_results.txt") in
    let oc = open_out ("results/" ^ txtout) in
    ignore (print_result oc res)
  | "vox" | "libri" | "vy" | "wsj" -> 
    let res, txtout = (match argv.(1) with
    | "vox" -> (find_words cmdlist accesstext_vox accesswav_vox dirpath true, 
        "vox_results.txt")
    | "libri" -> (find_words cmdlist accesstext_libri accessflac_libri dirpath 
        true, "libri_results.txt")
    | "vy" -> (find_words cmdlist accesstext_vy accesswav_vy dirpath false, 
      "vy_results.txt")
    | "wsj" -> print_endline "wsj"; (wsj0_dict dirpath cmdlist, "wsj_results.txt")
    | _ -> (D.empty,"")) in
    let cmd_dict = make_cmd_dict res D.empty in
    let oc = open_out ("results/" ^ txtout) in
    ignore (print_result oc cmd_dict)
  | _ -> failwith "argv 1 is invalid"
  ;;

main ()