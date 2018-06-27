exception Unimplemented

module type Formattable = sig
  type t
  val format : Format.formatter -> t -> unit
end

module type Comparable = sig
  type t
  val compare : t -> t -> [ `EQ | `GT | `LT ]
  include Formattable with type t := t
end

type ('k,'v) tree23 =
  | Leaf
  | Twonode of ('k,'v) twonode
  | Threenode of ('k,'v) threenode
and ('k,'v) twonode = {
  left2  : ('k,'v) tree23;
  value  : 'k * 'v;
  right2 : ('k,'v) tree23;
}
and ('k,'v) threenode = {
  left3   : ('k,'v) tree23;
  lvalue  : 'k * 'v;
  middle3 : ('k,'v) tree23;
  rvalue  : 'k * 'v;
  right3  : ('k,'v) tree23;
}

type ('k,'v) tree = 
  | Tree of ('k,'v) tree23
  | Upward of ('k,'v) tree23

module type Dictionary = sig
  module Key : Comparable
  module Value : Formattable
  type key = Key.t
  type value = Value.t
  type t
  val rep_ok : t  -> t
  val empty : t
  val is_empty : t -> bool
  val size : t -> int
  val insert : key -> value -> t -> t
  val member : key -> t -> bool
  val find : key -> t -> value option
  val remove : key -> t -> t
  val choose : t -> (key * value) option
  val fold : (key -> value -> 'acc -> 'acc) -> 'acc -> t -> 'acc
  val to_list : t -> (key * value) list
  val expose_tree : t -> (key,value) tree23
  val format : Format.formatter -> t -> unit
end

module type DictionaryMaker =
  functor (K : Comparable) (V : Formattable)
    -> Dictionary with module Key = K and module Value = V

module MakeListDictionary (K : Comparable) (V : Formattable) = struct
  module Key = K
  module Value = V
  type key = K.t
  type value = V.t

  (* AF: a [t] is an association list where the first value of each tuple is of 
   *     type [key] and the second value of each type is of type [value]. This 
   *     represents a dictionary that maps keys to values. 
   * RI: Each key must be unique, meaning each tuple in the list must have its
   *     first value be unique. *)
  type t = (key * value) list

  let rec rep_ok d =
    match d with 
    | [] -> d
    | h::t -> if List.mem h t then failwith "rep not okay" else rep_ok d

  let empty = []

  (* let compare d1 d2 = if d1 > d2 then 1 else if d1 < d2 then -1 else 0   *)

  let is_empty d =
    d = empty

  let size d =
    List.length d

  (* [compare_helper (k1,_) (k2,_)] compares the two keys k1 and k2 using 
   * Pervasives.compare and then translates that result to the local compare
   * function, thus either returning either `LT, `EQ, or `GT. 
   * requires: [k1] and [k2] are both valid keys of type Key.t *) 
   let compare_helper (k1,_) (k2,_) = 
    match Key.compare k1 k2 with 
    | `LT -> -1
    | `EQ -> 0
    | `GT -> 1

  let to_list d = 
    List.sort compare_helper d

  let rec fold (f:key -> value -> 'acc -> 'acc) init d =
    let f' acc (k,v) = f k v acc in 
    List.fold_left f' init (to_list d)

  let remove k d =
    let f key v acc = if K.compare key k = `EQ then acc else (key,v)::acc in
    fold f [] d

  let find k d = 
    let f key v acc = 
    if acc = None then 
      if (K.compare key k) = `EQ then (Some v )
      else None
    else acc in 
    fold f None d

  let member k d =
    let f key v acc = if K.compare key k = `EQ then true else (acc || false) in
    fold f false d 

  let insert k v d = 
    let d = if member k d then remove k d else d in
    (k, v)::d

  let choose d =
    match d with 
    | [] -> None
    | h::t -> Some h

  let expose_tree d =
    failwith "not a 2-3 tree"
    
  let rec format fmt d =
    let f x = match x with (k,v) -> 
      K.format fmt k; print_string ","; V.format fmt v; print_newline () in
    print_string "("; List.iter f (to_list d); print_endline ")";
end

module MakeTreeDictionary (K : Comparable) (V : Formattable) = struct
  module Key = K
  module Value = V
  type key = K.t
  type value = V.t

  (* AF: a [t] is an 2-3 tree that have keys and values associated with every 
   *     node of the tree. This represents a dictionary that maps keys to values 
   * RI: The depth of the subtrees of any certain tree must be 
   *     equal to each other. The values of a left subtree must be less than the
   *     values of parent node. The opposite is true for the right subtree. If
   *     a middle tree exists, all values in it must be between the 2 values.*)
  type t = (key,value) tree23

  (* [depth d] is the depth of the 2-3 tree d meaning the number of levels that
   * contain nodes in the tree
   * requires: [d] is a valid 2-3 tree (of type tree23 as defined) *)   
  let rec depth d = 
    match d with 
    | Leaf -> 0 
    | Twonode {left2;value;right2} -> 
      let l,r = depth left2, depth right2 in
      if l = r then 1 + l else failwith "rep not okay"
    | Threenode {left3;lvalue;middle3;rvalue;right3} -> 
      let l, m, r = depth left3, depth middle3, depth right3 in
      if l = r && l = m then 1 + m else failwith "rep not okay"
      
  let rep_ok d =
    match d with 
    | Leaf -> d
    | Twonode {left2;value;right2} -> 
      if depth left2 = depth right2 then d else failwith "rep not okay"
    | Threenode {left3;lvalue;middle3;rvalue;right3} -> 
      if depth left3 = depth right3 && depth left3 = depth middle3 then d else failwith "rep not okay"

  let empty = Leaf

  let is_empty d =
    d = Leaf

(** [size' d acc] is the size of tree [d], where [acc] is the accumulated size
  * of the whole tree so far. 
  * HELPER TO [size d]*)
  let rec size' d acc = 
    match d with 
    | Leaf -> acc
    | Twonode {left2;value;right2} -> size' left2 (size' right2 acc+1)
    | Threenode {left3;lvalue;middle3;rvalue;right3} -> 
            size' left3 (size' middle3 (size' right3 acc+2))

  let rec size d =
    size' d 0

(** [insert_h k v d] is the tree [d] after [k] and [v] are inserted in the 
  * proper location. 
  * requires: 
  *  - [d] is a valid 23tree with no pending upward nodes*)
  let rec insert_h k v d = 
    match d with
    | Leaf -> 
      let kv = (k,v) in
      Upward (Twonode {left2 = Leaf; value = kv; right2 = Leaf}) 
    | Twonode t -> let comp = K.compare k (fst t.value) in 
      if comp = `EQ then Tree (Twonode {t with value = (k,v)}) else 
      if comp = `LT then 
        let node = insert_h k v t.left2 in 
        (match node with 
          | Upward Twonode {left2=l;value=value;right2=r} -> 
            Tree (Threenode {left3=l;lvalue=value;middle3=r;rvalue=t.value;right3=t.right2}) 
          | Tree l -> Tree (Twonode {t with left2 = l}) 
          | _ -> failwith "Upwards are all Twonodes")
      else 
        let node = insert_h k v t.right2 in 
        (match node with 
          | Upward Twonode {left2=l;value=value;right2=r} -> 
            Tree (Threenode{left3=t.left2;lvalue=t.value;middle3=l;rvalue=value;right3=r})
          | Tree r -> Tree (Twonode {t with right2 = r}) 
          | _ -> failwith "Upwards are all Twonodes") 
    | Threenode t -> 
      let lcomp = K.compare k (fst t.lvalue) in
      let rcomp = K.compare k (fst t.rvalue) in 
      if lcomp = `EQ then Tree (Threenode {t with lvalue = (k,v)}) else 
      if rcomp = `EQ then Tree (Threenode {t with rvalue = (k,v)}) else       
      if lcomp = `LT then 
        let node = insert_h k v t.left3 in
        (match node with 
          | Upward l ->
            let right = Twonode {left2 = t.middle3; value=t.rvalue;right2=t.right3} in
            Upward (Twonode {left2=l;value=t.lvalue;right2=right})
          | Tree l -> Tree (Threenode {t with left3 = l}))
      else if rcomp = `GT then
        let node = insert_h k v t.right3 in
        (match node with 
          | Upward r -> 
            let left = Twonode{left2=t.left3;value=t.lvalue;right2=t.middle3} in
            Upward (Twonode {left2=left;value=t.rvalue;right2=r})
          | Tree r -> Tree (Threenode {t with right3 = r}))
      else
        let node = insert_h k v t.middle3 in
        (match node with 
          | Upward Twonode {left2=b; value=w; right2=c} -> 
            let left = Twonode {left2=t.left3;value=t.lvalue;right2=b} in
            let right = Twonode {left2=c;value=t.rvalue;right2=t.right3} in
            Upward (Twonode {left2=left;value=w;right2=right})
          | Tree m -> Tree (Threenode {t with middle3 = m})
          | _ -> failwith "Upwards are all Twonodes")

  let insert k v d =
    let t = insert_h k v d in
    match t with 
    | Tree t -> rep_ok t
    | Upward t -> rep_ok t 

(** [find_pred d] is the predecessor of the value of left subtree [d]. This is 
  * the value that replaces the value at the head. Calls itself on the rightmost 
  * node if [d] is not a terminal node. 
  * requires: 
  *  - [d] is a valid Tree with no holes, and fulfills the RI of 23trees. 
  *        Also, [d] is not a Leaf. *)
  let rec find_pred d = 
    match d with 
    | Leaf -> failwith "shouldn't be Leaf" 
    | Twonode t -> 
      if t.right2 = Leaf then t.value 
      else find_pred t.right2
    | Threenode t -> 
      if t.right3 = Leaf then t.rvalue
      else find_pred t.right3

(** [remove_h k d] is the Tree [d] after removing value with key [k]. May 
  * not contain any subtrees that are holes, but it itself may be a hole tree.
  * requires: 
  *  - [d] is a valid Tree. *)
  let rec remove_h k d =
    match d with
    | Leaf -> Tree d
    | Twonode t -> 
      if K.compare k (fst t.value) = `EQ && t.left2 = Leaf then Upward Leaf else 
      let v = if K.compare k (fst t.value) = `EQ then 
        find_pred t.left2 else t.value in
      if K.compare k (fst t.value) = `EQ || K.compare k (fst t.value) = `LT then 
      let node = if v = t.value then remove_h k t.left2 
        else remove_h (fst v) (t.left2) in 
      (match node with 
      | Tree l -> Tree (Twonode {t with left2=l}) 
      | Upward l -> 
        (match t.right2 with
        | Twonode {left2;value;right2} -> 
          Upward (Threenode 
          {left3=l;lvalue=v;middle3=left2;rvalue=value;right3=right2}) 
        | Threenode {left3=b;lvalue=y;middle3=c;rvalue=z;right3=d} -> 
          let left = Twonode {left2=l;value=v;right2=b} in
          let right = Twonode {left2=c;value=z;right2=d} in
          Tree (Twonode {left2=left;value=y;right2=right}) 
        | _ -> failwith "Stage4"))
      else 

      let node = remove_h k t.right2 in
      (match node with 
      | Tree r -> Tree (Twonode {t with right2=r}) 
      | Upward r -> 
        (match t.left2 with
        | Twonode {left2;value;right2} ->
          Upward (Threenode 
          {left3=left2;lvalue=value;middle3=right2;rvalue=v;right3=r})
        | Threenode {left3=a;lvalue=x;middle3=b;rvalue=y;right3=c} -> 
          let left = Twonode {left2=a;value=x;right2=b} in
          let right = Twonode {left2=c;value=v;right2=r} in
          Tree (Twonode {left2=left;value=y;right2=right}) 
        | _ -> failwith "Stage4"))

    | Threenode t -> 
      if t.left3 = Leaf && K.compare k (fst t.lvalue) = `EQ then 
        Tree (Twonode {value=t.rvalue;left2=Leaf;right2=Leaf}) else
      if t.left3 = Leaf && K.compare k (fst t.rvalue) = `EQ  then
        Tree (Twonode {value=t.lvalue;left2=Leaf;right2=Leaf}) else
      let lv = if K.compare k (fst t.lvalue) = `EQ 
        then find_pred t.left3 else t.lvalue in
      let rv = if K.compare k (fst t.rvalue) = `EQ 
        then find_pred t.middle3 else t.rvalue in
      if K.compare k (fst t.lvalue)=`EQ || K.compare k (fst t.lvalue)=`LT then 
        let node = if lv = t.lvalue then remove_h k t.left3 else 
          remove_h (fst lv) t.left3 in
        (match node with
        | Tree l -> Tree (Threenode {t with left3 = l})
        | Upward l -> 
          (match t.middle3 with
          | Twonode {left2=b;value=y;right2=c} -> 
            let left=Threenode{left3=l;lvalue=lv;middle3=b;rvalue=y;right3=c} in
            Tree (Twonode {left2=left;value=rv;right2=t.right3})
          | Threenode {left3=b;lvalue=x;middle3=c;rvalue=y;right3=d} -> 
            let left = Twonode {left2=l;value=lv;right2=b} in
            let middle = Twonode {left2=c;value=y;right2=d} in
            Tree (Threenode
              {left3=left;lvalue=x;middle3=middle;rvalue=rv;right3=t.right3})
          | Leaf -> failwith "Stage4"))
      else if K.compare k (fst t.rvalue) = `GT then
        let node = remove_h k t.right3 in
        (match node with
        | Tree r -> Tree (Threenode {t with right3 = r})
        | Upward r -> 
          (match t.middle3 with
          | Twonode {left2=b;value=y;right2=c} ->
            let right=Threenode{left3=b;lvalue=y;middle3=c;rvalue=rv;right3=r}in
            Tree (Twonode {left2=t.left3;value=lv;right2=right})
          | Threenode {left3=b;lvalue=x;middle3=c;rvalue=y;right3=d} -> 
            let middle = Twonode {left2=b;value=x;right2=c} in
            let right = Twonode {left2=d;value=rv;right2=r} in
            Tree (Threenode 
              {left3=t.left3;lvalue=lv;middle3=middle;rvalue=y;right3=right})
          | Leaf -> failwith "Stage4"))
      else 
      let node = if rv = t.rvalue then remove_h k t.middle3 else
        remove_h (fst rv) t.middle3 in
      (match node with 
      | Tree m -> Tree (Threenode {t with middle3 = m})
      | Upward m -> 
        (match t.left3 with
        | Twonode {left2=a;value=x;right2=b} -> 
          let left = Threenode {left3=a;lvalue=x;middle3=b;rvalue=lv;right3=m}in
          Tree (Twonode {left2=left;value=rv;right2=t.right3})
        | Threenode {left3=a;lvalue=w;middle3=b;rvalue=x;right3=c} ->
          let left = Twonode{left2=a;value=w;right2=b} in
          let middle = Twonode{left2=c;value=lv;right2=m} in
          Tree (Threenode
            {left3=left;middle3=middle;right3=t.right3;lvalue=x;rvalue=rv})
        | Leaf -> failwith "Stage4"))
  
  let remove k d = 
    match remove_h k d with
    | Tree t -> rep_ok t
    | Upward t -> rep_ok t

  let rec find k d =
    match d with 
    | Leaf -> None 
    | Twonode {left2;value=(x,y);right2} -> 
      if K.compare k x = `EQ then (Some y) else
      let n = if K.compare k x = `LT then find k left2 else find k right2 in 
      if n = None then None else n
    | Threenode {left3;lvalue=(lx,ly);middle3;rvalue=(rx,ry);right3} -> 
      if K.compare k lx = `EQ then Some ly else 
      if K.compare k rx = `EQ then Some ry else 
      let n = if K.compare k lx = `LT then find k left3 
      else if K.compare k rx = `GT then find k right3 else
      find k middle3 in 
      if n = None then None else n 
      
  let rec member k d =
    match d with 
    | Leaf -> false 
    | Twonode {left2;value=(x,_);right2} -> 
      if (K.compare k x = `EQ) then true else 
      let n = if (K.compare k x = `LT) then member k left2 else member k right2 in 
      (K.compare k x = `EQ) || n || false
    | Threenode {left3;lvalue=(lx,_);middle3;rvalue=(rx,_);right3} -> 
      if (K.compare k lx = `LT) || (K.compare k rx = `GT) then true else 
      let n = if (K.compare k lx = `LT) then member k left3 else 
        if (K.compare k rx = `GT) then member k right3 else
        member k middle3 in 
      k = lx || k = rx || n || false

  let choose d =
    match d with 
    | Leaf -> None
    | Twonode t -> Some t.value
    | Threenode t -> Some t.lvalue

  (* [fast_cat l1 l2] concats two lists, [l1] and [l2] together and returns 
   * that concatenated list. 
   * requires: [l1] and [l2] are both 'a lists of default type 'a list *) 
  let rec fast_cat l1 l2 = 
    let f acc e = e :: acc in
    List.rev (List.fold_left f (List.rev l1) l2 )

  let rec to_list d =
    match d with
    Leaf -> [] 
    | Twonode {left2;value;right2} -> 
      let l = to_list left2 in 
      let r = to_list right2 in 
      fast_cat l (value::r)
    | Threenode {left3;lvalue;middle3;rvalue;right3} -> 
      let l = to_list left3 in 
      let m = to_list middle3 in 
      let r = to_list right3 in
      fast_cat (fast_cat l (lvalue::m)) (rvalue::r)

  let expose_tree d =
    d

  let fold f init d =
    let f' acc (k,v) = f k v acc in 
    List.fold_left f' init (to_list d)

  let format fmt d =
    let f x = match x with (k,v) -> 
      K.format fmt k; print_string ","; V.format fmt v; print_newline () in
    print_string "("; List.iter f (to_list d); print_endline ")"

end

module type Set = sig
  module Elt : Comparable
  type elt = Elt.t
  type t
  val rep_ok : t  -> t
  val empty : t
  val is_empty : t -> bool
  val size : t -> int
  val insert : elt -> t -> t
  val member : elt -> t -> bool
  val remove : elt -> t -> t
  val union : t -> t -> t
  val intersect : t -> t -> t
  val difference : t -> t -> t
  val choose : t -> elt option
  val fold : (elt -> 'acc -> 'acc) -> 'acc -> t -> 'acc
  val to_list : t -> elt list
  val format : Format.formatter -> t -> unit
end

module MakeSetOfDictionary (C : Comparable) (DM:DictionaryMaker) = struct
  module Elt = C
  type elt = Elt.t
  module D = DM (Elt)(Elt)

  (* AF: a [t] is an dictionary that maps keys to values. We only care about 
   *     the keys for our sets, so the values in this dictionary type are 
   *     ignored/not important. The combination of all the keys in the dict. 
   *     is what represents this set. 
   * RI: Each key must be unique, meaning each key in the dict must have no
   *     other keys equal to it. This allows for the Set to have no
   *     duplicate elements. *)
  type t = D.t

  let empty = 
    D.empty

  let is_empty s =
    D.is_empty s

  let size s =
    D.(size s)

  let insert x s =
    D.insert x x s

  let member x s =
    D.member x s

  let remove x s =
    D.remove x s
  
  (* [getKey kv] takes in a (k,v) option and returns Some k if the option is  
   * not None and returns None if the option is indeed None. 
   * requires: [kv] is of type (k,v) option *) 
  let getKey kv = 
    match kv with
    | None -> None
    | Some (k, _) -> Some k

  (* [get_key_no_option kvo] takes in a (k,v) option and returns k if the   
   * option is not None and failswith "No key present" if it is None as there 
   * would be nothing to return from the option 
   * requires: [kvo] is of type (k,v) option *) 
  let get_key_no_option (kvo : ('a * 'b) option) = 
    match kvo with
    | None -> failwith "No key present"
    | Some (k,_) -> k

  let choose s =
    getKey (D.choose s)

  let fold f init s =
    let g k v acc = f k acc in
    D.fold g init s

  let rec union s1 s2 =
    let f k v acc = if member k acc then acc else insert k acc in
    D.fold f s1 s2
  
  (* [intersect' s1 s2 acc] takes in two sets and iterates over the first set. 
   * If any element of that first set is present in the second set, that elmt 
   * is added to an accumulator. The accumulator is eventually returned, 
   * meaning the function returns the intersection of the two sets. 
   * requires: [s1] and [s2] are sets of type D.t
   * [acc] is also a set of type D.t *) 
  let rec intersect' s1 s2 (acc: D.t) = 
    if D.is_empty s1 || D.is_empty s2 then acc else 
    let k = get_key_no_option (D.choose s1) in 
    if D.member k s2 then intersect' (D.remove k s1) (s2) (D.insert k k acc) 
    else intersect' (D.remove k s1) (s2) (acc)
  
  let intersect s1 s2 =
    intersect' s1 s2 D.empty

  let rec format fmt d =
    let f k v acc = print_string "("; 
      C.format fmt k; print_string ")"; in
    D.fold f () d

  (* [difference' s1 s2 acc] takes in two sets and iterates over the first set. 
   * If any element of that first set is present in the second set, that elmt 
   * is not added to an accumulator. The accumulator is eventually returned, 
   * meaning the function returns the difference of the two sets. 
   * requires: [s1] and [s2] are sets of type D.t
   * [acc] is also a set of type D.t *) 
  let rec difference' s1 s2 acc = 
    (* remove everything in s1 that is in s2 *)
    if D.is_empty s1 || D.is_empty s2 then acc else 
    let k = get_key_no_option (D.choose s1) in 
    if (D.member k s2) then 
    difference' (D.remove k s1) (s2) (acc) else 
    difference' (D.remove k s1) (s2) (D.insert k k acc)

  let rec difference s1 s2 =
    (* remove everything in s1 that is in s2 *)
    difference' s1 s2 D.empty

  let to_list s =
    D.to_list s |> 
    List.split |>
    fst

  let rec rep_ok s =
    let f k v acc = if D.member k acc then acc else D.insert k k acc in 
    let perfect_set = D.fold f empty s in  
    if perfect_set = s then s else failwith "rep not ok"


end
