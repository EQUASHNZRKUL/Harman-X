
exception IncompatibleSizeError

module type Matrix = sig
  type elt
  type t = elt list list
  val rep_ok : t -> t
  val empty : t
  val is_empty : t -> bool
  val size : t -> int
  val add : elt -> t -> t
  val remove : elt -> t -> t
  val insert : t -> int -> int -> elt -> t
  val insert_col : t -> int -> elt list -> t
  val elt : t -> int -> int -> elt
  val col : t -> int -> elt list
  val map : ('a -> 'b) -> t -> t
end

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

(** fft.ml --- Cooley-Tukey fast Fourier transform algorithm *)

  open Format
  open Complex

  (** [get_n_bits n] returns the number of bits of [n]. *)
  let get_n_bits =
    let rec aux n i =
      if i = 1 then n
      else if i > 0 && i land 1 = 0 then aux (n + 1) (i lsr 1)
      else invalid_arg "invalid input length"
    in
    aux 0

  (** [bitrev n i] bit-reverses [n]-digit integer [i]. *)
  let bitrev =
    let rec aux acc n i =
      if n = 0 then acc else aux ((acc lsl 1) lor (i land 1)) (n - 1) (i lsr 1)
    in
    aux 0

  let ( +! ) = add
  let ( -! ) = sub
  let ( *! ) = mul
  let ( /! ) = div

  let make_twiddle_factors len =
    let pi = 3.14159265358979 in
    let c = ~-. 2.0 *. pi /. float len in
    Array.init (len / 2) (fun i -> exp { re = 0.; im = c *. float i })

  let fft x =
    let len = List.length x in
    let n_bits = get_n_bits len in
    let w = make_twiddle_factors len in
    let y = Array.init len (fun i -> List.nth x (bitrev n_bits i)) in
    let butterfly m n ofs =
      for i = 0 to n / 2 - 1 do
        let j, k = ofs + i, ofs + i + n / 2 in
        let a, b = y.(j), y.(k) in
        y.(j) <- a +! w.(i * m) *! b;
        y.(k) <- a -! w.(i * m) *! b;
      done
    in
    for nb = 1 to n_bits do
      let n = 1 lsl nb in
      let m = 1 lsl (n_bits - nb) in
      for i = 0 to m - 1 do butterfly m n (n * i) done
    done;
    y


(** sigproc.ml - Signal Processing functions**)

  let (--) i j = 
    let rec ran n acc = 
      if n < i then acc else ran (n-1) (n::acc) 
    in ran j []

  (** [arange i j s] is the list of every [s] numbers from [i] to [j] exclusive *)
  let arange i j s = 
    let rec ran n acc = 
      if n < i then acc else ran (n-s) (n::acc)
    in ran (j-1) []

  (** [tile l x acc] is a list of [x] l times. *)
  let rec tile l x acc = if l > 0 then tile (l-1) x (x::acc) else acc

  let zeros l acc = tile l 0 acc

  let rec transpose matrix = 
    match matrix with
    | [] -> []
    | [] :: xss -> transpose xss
    | (x::xs)::xss -> 
      (x::List.map List.hd xss) :: transpose (xs :: List.map List.tl xss)

  (** [matrix_add A B acc] is the sum of matrices A and B. A and B must be the
    * same shape. *)
  let rec matrix_add A B acc = 
    match A,B with
    | [],[] -> acc
    | (ha::ta), (hb::tb) -> matrix_add ta tb ((List.map2 (+) ha hb)::acc)
    | _, _ -> raise IncompatibleSizeError

  let rec matrix_op A B op acc = 
    match A,B with
    | [],[] -> acc
    | (ha::ta), (hb::tb) -> matrix_op ta tb op ((List.map2 op ha hb)::acc)
    | _, _ -> raise IncompatibleSizeError

  (** [matrix_unop A op acc] is the result of executing [op] on all elts of [a] *)
  let rec matrix_unop A op acc = 
    match A with
    | [] -> acc
    | (h::t) -> matrix_unop t op ((List.map op h)::acc)

  let preemphasis audiosignal = 
    let signal = List.tl audiosignal.signal in
    let h = List.hd audiosignal.signal in
    let rec f i acc = 
      if i < List.length signal then f (i+1)
        (((List.nth signal i) -. audiosignal.preemph *. 
        (List.nth signal (List.length signal - i)))::acc) else acc in
    let signal = f 0 [] in
    h :: signal

  let rec sublist lst len acc = 
    match lst with
    | [] -> raise (Invalid_argument "lst length too short")
    | h::t when len > 0 -> sublist t (len-1) (List.rev (h::(List.rev acc)))
    | _ -> acc

  let remap alst bmatr = 
    let bl = List.length bmatr in
    let bh = List.length (List.hd bmatr) in
    let aseg = sublist alst bh [] in
    tile bl aseg []

  let framesig signal flen fstep winfunc = 
    let signal = List.map int_of_float signal in
    let siglen = List.length signal in
    let flen = int_of_float (ceil flen) in
    let fstep = int_of_float (ceil fstep) in
    let numframes = (if siglen <= flen then 1 else 
      1 + int_of_float (ceil (float_of_int (siglen - flen) /. float_of_int fstep))) in
    let padded_len = ((numframes - 1) * fstep + flen) in
    let zero_lst = zeros (padded_len - siglen) [] in
    let padded_sig = signal @ zero_lst in
      let aindices = tile numframes (arange 0 flen 1) [] in
      let bindices = tile flen (arange 0 (numframes * fstep) fstep) [] in
    let indices = matrix_op aindices (transpose bindices) (+) [[]] in
    let frames = remap padded_sig indices in
    let win_output = winfunc flen in
    let wins = tile numframes win_output [] in
    matrix_op frames wins ( * ) [[]]

  let magspec frames = 
    let complex_lst col = List.map (fun x -> polar x 0.0) col in
    let complex_spec = List.map (fun col -> fft (complex_lst col)) frames in
    List.map (fun col -> Array.to_list (Array.map (norm) col)) complex_spec

  let powspec frames nfft = 
    let frames' = magspec frames in
    let f col = List.map (fun x -> 1.0 /. (nfft *. x *. x)) col in
    List.map f frames'

  let hz2mel hz = 2595. *. (log10 (1. +. hz/.700.))

  let mel2hz mel = 700. *. (10. ** (mel/.2595.)-.1.)

  let rec ones acc x = 
    match x with
    | 1 -> 1::acc
    | _ -> ones (1::acc) (x-1)

(* Base.ml - Main Driver functions *)

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
    let frames = framesig signal (audiosignal.winlen *. audiosignal.samplerate) 
                (audiosignal.winlen *. audiosignal.samplerate) audiosignal.winfunc in
    let powspec = powspec (matrix_unop frames float_of_int [[]]) 
                  (float_of_int audiosignal.nfft) in
  