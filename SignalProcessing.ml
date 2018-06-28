

let hz2mel hz = 2595. *. (log10 (1. +. hz/.700.))

let mel2hz mel = 700. *. (10. ** (mel/.2595.)-.1.)

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
  preemph : bool;
  ceplifter : int;
  appendEnergy : bool;
  winfunc : int -> int list
  }

let rec ones acc x = 
  match x with
  | 1 -> 1::acc
  | _ -> ones (1::acc) (x-1)

let init_signal signal ?samplerate:(sr=16000.) ?winlen:(wl=0.025) ?winstep:(ws=0.01)
 ?numcep:(ncep=13) ?nfilt:(nflt=26) ?nfft:(nft=512) ?lowfreq:(lf=0.) 
 ?highfreq:(hf=None) ?preemph:(prmf=true) ?ceplifter:(cpl=22) 
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
let fbank signal =
  let highfreq = (match highfreq with | None -> samplerate/.2.0 | Some x -> x) in
  let signal = 