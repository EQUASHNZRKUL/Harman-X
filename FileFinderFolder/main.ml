open Filefinder
open Str
open Sys
open Data

let main () = 
  let simpleton = fun x y z -> x in
  let args = Sys.argv in
  let cmdlist = getCmdList argv.(1) [] in
  let dirpath = if argv.(2) = "" then "./FileFinderData" else argv.(2) in

let main () = 
  let simpleton = fun x y z -> x in
  let args = Sys.argv in
  let cmdlist = getCmdList argv.(2) [] in
  let dirpath = if argv.(3) = "" then "./FileFinderData" else argv.(2) in
  if argv.(3) = "export" then 
    let results = total_res_dict "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/" in
    let trash = dir_accumulate results "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/data/" in
    let trash = dir_accumulate_merged results "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/results/data_merged/" cmdlist in
    ignore(trash)
  else
  (* let taccess = accesstext_maker args.(3) args.(4) in *)
  let waccess = accesswav_maker args.(4) args.(6) in
  let res,txtout = (match argv.(1) with 
    | "vox" -> (find_words cmdlist accesstext_vox accesswav_vox dirpath true, 
                "vox_results.txt")
    | "libri" -> (find_words cmdlist accesstext_libri accessflac_libri dirpath 
                true, "libri_results.txt")
    | "surf" -> (surf_dict dirpath cmdlist, "surf_results.txt")
    | "vy" -> (find_words cmdlist accesstext_vy accesswav_vy dirpath false, 
              "vy_results.txt")
    | "ami" -> (D.empty, "metadata.ami_results.txt")
    | "wsj" -> print_endline "wsj"; (wsj0_dict dirpath cmdlist, "wsj_results.txt")
    | _ -> (D.empty,"")) in
  let cmd_dict = make_cmd_dict res D.empty in
  let oc = open_out ("results/" ^ txtout) in
  if argv.(1) = "ami" then 
    (ignore(print_result_ami oc (ami_dict dirpath cmdlist));
    close_out oc;
    let oc = open_out ("results/ami_results.txt") in
    ami_textify "results/metadata.ami_results.txt" oc)
  else if argv.(1) = "surf" then 
    ignore (print_result oc res)
  else ignore (print_result oc cmd_dict);
  close_out oc;
  (* ami_textify "results/metadata.ami_results.txt" (open_out "results/ami_results.txt") *)
  (* txtify "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/WSJ0" *)
  (* wsj0_unbox "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/WSJ0_meta/wsjdt/wsj0" *)
  (* flatten "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/LibriSpeech_360/train-clean-360" *)
  (* unflatten "/Users/justinkae/Documents/TensorFlowPractice/FileFinderFolder/FileFinderData/Vystidial/data/" *)
  ;;

main ()