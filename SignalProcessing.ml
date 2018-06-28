
type audiosignal = {
  signal : float list;
  samplerate : float;
  winlen : float;
  winstep : float;
  numcep : int;
  nfilt : int;
  nfft : int;
  lowfreq : float;
  highfreq : float option;
  preemph : float;
  ceplifter : int;
  appendEnergy : bool;
  winfunc : int -> int list
}

let (--) i j = 
  let rec ran n acc = 
    if n < i then acc else ran (n-1) (n::acc) 
  in ran j []

let arange i j s = 
  let rec ran n acc = 
    if n < i then acc else ran (n-s) (n::acc)
  in ran (j-1) []

let zeros l acc = tile l 0 acc

let rec tile l x acc = if l > 0 then tile (l-1) x (x::acc) else acc

let preemphasis audiosignal = 
  let signal = List.tl audiosignal.signal in
  let h = List.hd audiosignal.signal in
  let rec f i acc = 
    if i < List.length signal then f (i+1)
      (((List.nth signal i) -. audiosignal.preemph *. 
      (List.nth signal (List.length signal - i)))::acc) else acc in
  let signal = f 0 [] in
  h :: signal

let framesig signal flen fstep winfunc = 
  let siglen = List.length signal in
  let flen = int_of_float (ceil flen) in
  let fstep = int_of_float (ceil fstep) in
  let numframes = (if siglen <= flen then 1 else 
    1 + int_of_float (ceil (float_of_int (siglen - flen) /. float_of_int fstep))) in
  let padded_len = (numframes - 1) * fstep + flen in
  let zeros = zeros (padded_len - siglen) [] in
  let padded_sig = signal @ zeros in 
  let indices = tile()
  

let hz2mel hz = 2595. *. (log10 (1. +. hz/.700.))

let mel2hz mel = 700. *. (10. ** (mel/.2595.)-.1.)

let rec ones acc x = 
  match x with
  | 1 -> 1::acc
  | _ -> ones (1::acc) (x-1)

let init_signal signal ?samplerate:(sr=16000.) ?winlen:(wl=0.025) ?winstep:(ws=0.01)
 ?numcep:(ncep=13) ?nfilt:(nflt=26) ?nfft:(nft=512) ?lowfreq:(lf=0.) 
 ?highfreq:(hf=None) ?preemph:(prmf=0) ?ceplifter:(cpl=22) 
 ?appendEnergy:(appe=true) ?winfunc:(wf=ones []) = 
 {
   signal = signal;
   samplerate = sr;
   winlen = wl;
   winstep = ws;
   numcep = ncep;
   nfilt = nflt;
   nfft = nft;
   lowfreq = lf;
   highfreq = hf;
   preemph = prmf;
   ceplifter = cpl;
   appendEnergy = appe;
   winfunc = wf
 }


(** [fbank signal ...] is the list of mel-filterbank energy features from an 
  * audio signal [signal]. [highfreq] is a float opt that either gives the upper
  * bound of the filters or None indicating its samplerate/2. 
*)
let fbank audiosignal =
  let highfreq = 
    (match audiosignal.highfreq with 
    | None -> audiosignal.samplerate/.2.0 | Some x -> x) in
  let signal = preemphasis audiosignal in
  let frames = framesig signal (signal.winlen * signal.samplerate) 
               (signal.winlen * signal.samplerate) winfunc
  let powspec = 
  