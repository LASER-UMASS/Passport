(**************************************************************)
(*   Copyright Dominique Larchey-Wendling [*]                 *)
(*                                                            *)
(*                             [*] Affiliation LORIA -- CNRS  *)
(**************************************************************)
(*      This file is distributed under the terms of the       *)
(*         CeCILL v2 FREE SOFTWARE LICENSE AGREEMENT          *)
(**************************************************************)

Require Import List Permutation Arith Omega.

Require Import utils pos vec.

Set Implicit Arguments.

(** * Intuionistic Linear Logic *)

Local Infix "~p" := (@Permutation _) (at level 70).

Definition ll_vars := nat.

(** We only consider the fragment of ILL which
   contains !, -o and & ... 
 
   ILL can be faithfully embedded into that fragment 
   but this does not matter for undecidability 
*)

Inductive ll_conn := ll_with | ll_limp | ll_times | ll_plus.
Inductive ll_cst := ll_one | ll_bot | ll_top.

Inductive ll_form : Set :=
  | ll_var  : ll_vars -> ll_form
  | ll_zero : ll_cst  -> ll_form
  | ll_ban  : ll_form -> ll_form
  | ll_bin  : ll_conn -> ll_form -> ll_form -> ll_form.

(* Symbols for cut&paste â   â   đ  īš  â  â  â¸  â   âŧ  â  âĸ *)

Notation "â" := (ll_zero ll_top).
Notation "â" := (ll_zero ll_bot).
Notation đ := (ll_zero ll_one).

Infix "&" := (ll_bin ll_with) (at level 50, only parsing).
Infix "īš " := (ll_bin ll_with) (at level 50).
Infix "â" := (ll_bin ll_times) (at level 50).
Infix "â" := (ll_bin ll_plus) (at level 50).
Infix "-o" := (ll_bin ll_limp) (at level 51, only parsing, right associativity).
Infix "â¸" := (ll_bin ll_limp) (at level 51, right associativity).

Notation "'!' x" := (ll_ban x) (at level 52, only parsing).
Notation "â x" := (ll_ban x) (at level 52).

Notation "ÂŖ" := ll_var.

Definition ll_lbang := map (fun x => !x).

Notation "'!l' x" := (ll_lbang x) (at level 60, only parsing).
Notation "âŧ x" := (ll_lbang x) (at level 60).

Notation "â" := nil (only parsing).

Reserved Notation "l 'âĸ' x" (at level 70, no associativity).

Inductive S_ill : list ll_form -> ll_form -> Prop :=

  (* These are the SILL rules in the paper *)

  | in_llp_ax     : forall A,                         A::â âĸ A

  | in_llp_perm   : forall Î Î A,              Î ~p Î     ->   Î âĸ A 
                                           (*-----------------------------*)
                                      ->                 Î âĸ A

  | in_llp_limp_l : forall Î Î A B C,          Î âĸ A      ->   B::Î âĸ C
                                           (*-----------------------------*)    
                                      ->           A â¸ B::Î++Î âĸ C

  | in_llp_limp_r : forall Î A B,                    A::Î âĸ B
                                           (*-----------------------------*)
                                      ->            Î âĸ A â¸ B

  | in_llp_with_l1 : forall Î A B C,                  A::Î âĸ C 
                                           (*-----------------------------*)
                                      ->           Aīš B::Î âĸ C

  | in_llp_with_l2 : forall Î A B C,                  B::Î âĸ C 
                                           (*-----------------------------*)
                                      ->           Aīš B::Î âĸ C
 
  | in_llp_with_r : forall Î A B,               Î âĸ A     ->   Î âĸ B
                                           (*-----------------------------*)
                                      ->              Î âĸ Aīš B

  | in_llp_bang_l : forall Î A B,                    A::Î âĸ B
                                           (*-----------------------------*)
                                      ->            â A::Î âĸ B

  | in_llp_bang_r : forall Î A,                       âŧÎ âĸ A
                                           (*-----------------------------*)
                                      ->              âŧÎ âĸ â A

  | in_llp_weak : forall Î A B,                        Î âĸ B
                                           (*-----------------------------*)
                                      ->           â A::Î âĸ B

  | in_llp_cntr : forall Î A B,                    â A::â A::Î âĸ B
                                           (*-----------------------------*)
                                      ->             â A::Î âĸ B

  (* These are the other rule for a complete sequent calculus for the whole ILL *)

  | in_llp_cut : forall Î Î A B,                 Î âĸ A    ->   A::Î âĸ B
                                           (*-----------------------------*)    
                                      ->              Î++Î âĸ B

  | in_llp_times_l : forall Î A B C,               A::B::Î âĸ C 
                                           (*-----------------------------*)
                                      ->            AâB::Î âĸ C
 
  | in_llp_times_r : forall Î Î A B,             Î âĸ A    ->   Î âĸ B
                                           (*-----------------------------*)
                                      ->              Î++Î âĸ AâB

  | in_llp_plus_l :  forall Î A B C,            A::Î âĸ C  ->  B::Î âĸ C 
                                           (*-----------------------------*)
                                      ->            AâB::Î âĸ C

  | in_llp_plus_r1 : forall Î A B,                    Î âĸ A  
                                           (*-----------------------------*)
                                      ->              Î âĸ AâB

  | in_llp_plus_r2 : forall Î A B,                    Î âĸ B  
                                           (*-----------------------------*)
                                      ->              Î âĸ AâB

  | in_llp_bot_l : forall Î A,                     â::Î âĸ A

  | in_llp_top_r : forall Î,                          Î âĸ â

  | in_llp_unit_l : forall Î A,                       Î âĸ A  
                                           (*-----------------------------*)
                                      ->           đ ::Î âĸ A

  | in_llp_unit_r :                                   â âĸ đ

where "l âĸ x" := (S_ill l x).

Definition ILL_PROVABILITY (c : (list ll_form) * ll_form) := let (Ga,A) := c in Ga âĸ A. 

Fact S_ill_weak Î Î B : Î âĸ B -> âŧÎ++Î âĸ B.
Proof. intro; induction Î; simpl; auto; apply in_llp_weak; auto. Qed.

Fact S_ill_cntr Î Î B : âŧÎ++âŧÎ++Î âĸ B -> âŧÎ++ Î âĸ B.
Proof.
  revert Î Î; intros Ga.
  induction Ga as [ | A Ga IH ]; simpl; auto; intros De.
  intros H.
  apply in_llp_cntr.
  apply in_llp_perm with (map ll_ban Ga ++ (!A::!A::De)).
  + apply Permutation_sym.
    do 2 apply Permutation_cons_app; auto.
  + apply IH.
    revert H; apply in_llp_perm.
    rewrite app_assoc.
    apply Permutation_cons_app.
    rewrite <- app_assoc.
    apply Permutation_app; auto.
    apply Permutation_cons_app; auto.
Qed.

Theorem S_ill_weak_cntr ÎŖ Î A B : In A ÎŖ -> âŧÎŖ++Î âĸ B <-> â A::âŧÎŖ++Î âĸ B.
Proof.
  revert ÎŖ Î; intros Si Ga.
  intros H.
  apply In_perm in H.
  destruct H as (Si' & H).
  split.
  + apply in_llp_weak.
  + intros H1.
    apply in_llp_perm with (âŧ(A :: Si') ++ Ga).
    * apply Permutation_app; auto.
      apply Permutation_map; auto.
    * simpl; apply in_llp_cntr.
      revert H1; apply in_llp_perm.
      simpl; apply Permutation_cons; auto.
      change (â A::âŧSi'++Ga) with (âŧ(A::Si')++Ga).
      apply Permutation_app; auto.
      apply Permutation_map, Permutation_sym; auto.
Qed.

Section trivial_phase_semantics.

  Variables (n : nat) (s : ll_vars -> vec nat n -> Prop).

  Reserved Notation "'âĻ' A 'â§'" (at level 65).

  Definition ll_tps_imp (X Y : _ -> Prop) (v : vec _ n) := forall x, X x -> Y (vec_plus x v).
  Definition ll_tps_mult (X Y : _ -> Prop) (x : vec _ n) := exists a b, x = vec_plus a b /\ X a /\ Y b. 
  
  Infix "**" := (ll_tps_mult) (at level 65, right associativity).
  Infix "-*" := (ll_tps_imp) (at level 65, right associativity).

  Fact ll_tps_mult_mono (X1 X2 Y1 Y2 : _ -> Prop) : 
             (forall x, X1 x -> X2 x)
          -> (forall x, Y1 x -> Y2 x)
          -> (forall x, (X1**Y1) x -> (X2**Y2) x).
  Proof.
    intros H1 H2 x (a & b & H3 & H4 & H5); subst.
    exists a, b; auto.
  Qed.

  (* Symbols for cut&paste â   â   đ  īš  â  â  â¸  â   âŧ  â  âĸ âĻ â§ Î Î ÎŖ*)

  Fixpoint ll_tps A x : Prop :=
    match A with
      | ÂŖ X     => s X x
      | A īš  B  => âĻAâ§ x /\ âĻBâ§ x
      | â A     => âĻAâ§ x /\ x = vec_zero
      | A â¸ B  => (âĻAâ§ -* âĻBâ§) x
      | A â B  => (âĻAâ§ ** âĻBâ§) x
      | A â B  => âĻAâ§ x \/ âĻBâ§ x
      | â             => False
      | â             => True
      | đ              => x = vec_zero
    end
  where "âĻ A â§" := (ll_tps A).

  Reserved Notation "'[[' Î ']]'" (at level 0).

  Fixpoint ll_tps_list Î :=
    match Î with
      | â    => eq vec_zero
      | A::Î => âĻAâ§ ** [[Î]]
    end
  where "[[ Î ]]" := (ll_tps_list Î).

  Fact ll_tps_app Î Î x : [[Î++Î]] x <-> ([[Î]]**[[Î]]) x.
  Proof.
    revert Î Î x; intros Ga De.
    induction Ga as [ | A Ga IH ]; intros x; simpl; split; intros Hx.
    + exists vec_zero, x; simpl; rew vec.
    + destruct Hx as (a & b & H1 & H2 & H3); subst; auto; rewrite vec_zero_plus; auto.
    + destruct Hx as (a & b & H1 & H2 & H3).
      apply IH in H3.
      destruct H3 as (c & d & H4 & H5 & H6).
      exists (vec_plus a c), d; split.
      * subst; apply vec_plus_assoc.
      * split; auto.
        exists a, c; auto.
    + destruct Hx as (y & d & H1 & H2 & H3).
      destruct H2 as (a & g & H2 & H4 & H5).
      exists a, (vec_plus g d); split.
      * subst; symmetry; apply vec_plus_assoc.
      * split; auto.
        apply IH.
        exists g, d; auto.
  Qed.

  Fact ll_tps_lbang Î x : [[âŧÎ]] x <-> [[Î]] x /\ x = vec_zero.
  Proof.
    revert Î x; intros Ga.
    induction Ga as [ | A Ga IH ]; intros x.
    + simpl; split; auto; tauto.
    + split.
      * intros (a & g & H1 & H2 & H3).
        split.
        - exists a, g; repeat split; auto.
          ** apply H2.
          ** apply IH; auto.
        - apply IH, proj2 in H3.
          apply proj2 in H2; subst; auto.
          apply vec_zero_plus.
      * intros ((a & g & H1 & H2 & H3) & H4).
        exists x, x.
        assert (a = vec_zero /\ g = vec_zero) as E.
        { apply vec_plus_is_zero; subst; auto. }
        destruct E; subst; repeat split; auto; rew vec.
        apply IH; auto.
  Qed.

  Fact ll_tps_list_bang_zero Î : [[âŧÎ]] vec_zero <-> forall A, In A Î -> âĻAâ§ vec_zero.
  Proof.
    induction Î as [ | A Ga IH ].
    + split.
      * simpl; tauto.
      * intros _; simpl; auto.
    + split.
      * intros (u & v & H1 & H2 & H3).
        destruct H2 as [ H2 H4 ]; subst; auto.
        rewrite vec_zero_plus in H1; subst.
        rewrite IH in H3.
        intros B [ E | HB ]; subst; auto.
      * intros H.
        exists vec_zero, vec_zero.
        rewrite vec_zero_plus; repeat split; auto.
        - apply H; left; auto.
        - rewrite IH.
          intros; apply H; right; auto.
  Qed.

  (* Symbols for cut&paste â   â   đ  īš  â  â  â¸  â   âŧ  â  âĸ âĻ â§ Î Î ÎŖ*)
 
  Fact ll_tps_perm Î Î : Î ~p Î -> forall x, [[Î]] x -> [[Î]] x.
  Proof.
    induction 1 as [ | A Ga De H IH | A B Ga | ]; simpl; auto.
    + intros x (a & b & H1 & H2 & H3).
      exists a, b; repeat split; auto.
    + intros x (a & b & H1 & H2 & c & d & H3 & H4 & H5).
      exists c, (vec_plus a d); split.
      * subst; rewrite (vec_plus_comm c), vec_plus_assoc, (vec_plus_comm c); auto.
      * split; auto.
        exists a, d; auto.
  Qed.
  
  Definition ll_sequent_tps Î A := [[Î]] -* âĻAâ§.

  Notation "'[<' Î '|-' A '>]'" := (ll_sequent_tps Î A).

  Fact ll_sequent_tps_mono Î A B :
     (forall x, âĻAâ§ x -> âĻBâ§ x) -> forall x, [< Î |- A >] x -> [< Î |- B >] x.
  Proof.
    intros H x; simpl; unfold ll_sequent_tps.
    intros H1 a H2.
    apply H, H1; auto.
  Qed.

  Fact ll_perm_tps Î Î : Î ~p Î -> forall x A, [< Î |- A >] x -> [< Î |- A >] x.
  Proof.
    intros H1 x B; unfold ll_sequent_tps.
    intros H2 a H3.
    apply H2; revert H3. 
    apply ll_tps_perm, Permutation_sym; auto.
  Qed.

  Fact ll_sequent_tps_eq  Î A : [< Î |- A >] vec_zero <-> forall x, [[Î]] x -> âĻAâ§ x.
  Proof.
    split.
    * intros H x Hx.
      rewrite <- vec_zero_plus, vec_plus_comm.
      apply (H x); trivial.
    * intros H x Hx.
      rewrite vec_plus_comm, vec_zero_plus; auto.
  Qed.

  Theorem ll_tps_sound Î A : Î âĸ A -> [< Î |- A >] vec_zero.
  Proof.
    induction 1 as [ A 
                   | Ga De A H1 H2 IH2
                   | Ga De A B C H1 IH1 H2 IH2
                   | Ga A B H1 IH1
                   | Ga A B C H1 IH1
                   | Ga A B C H1 IH1
                   | Ga A B H1 IH1 H2 IH2
                   | Ga A B H1 IH1 
                   | Ga A H1 IH1
                   | Ga A B H1 IH1
                   | Ga A B H1 IH1

                   | Ga De A B H1 IH1 H2 IH2
                   | Ga A B C H1 IH1
                   | Ga De A B H1 IH1 H2 IH2
                   | Ga A B C H1 IH1 H2 IH2
                   | Ga A B H1 IH1
                   | Ga A B H1 IH1
                   | Ga A
                   | Ga
                   | Ga A H1 IH1
                   |
                   ]; unfold ll_sequent_tps in * |- *.

    + intros x; simpl; intros (a & b & H1 & H2 & H3); subst; eq goal H2.
      f_equal; do 2 rewrite vec_plus_comm, vec_zero_plus; auto.

    + revert IH2; apply ll_perm_tps; auto.

    + intros x (y & z & H3 & H4 & H5); simpl.
      apply IH2.
      apply ll_tps_app in H5.
      destruct H5 as (g & d & H5 & H6 & H7).
      simpl in H4.
      apply IH1, H4 in H6.
      exists (vec_plus y g), d; repeat split; auto.
      * subst; apply vec_plus_assoc.
      * eq goal H6; f_equal; rew vec.

    + simpl; intros y Hy a Ha.
      rewrite vec_plus_assoc.
      apply IH1.
      exists a, y; repeat split; auto; omega.

    + intros x (a & b & H2 & H3 & H4); apply IH1.
      exists a, b; repeat split; auto; apply H3.

    + intros x (a & b & H2 & H3 & H4); apply IH1.
      exists a, b; repeat split; auto; apply H3.

    + intros x Hx; split.
      * apply IH1; auto.
      * apply IH2; auto.
   
    + intros x (a & g & H2 & H3 & H4).
      apply IH1; exists a, g; repeat split; auto.
      apply H3.

    + intros x Hx; split.
      apply IH1; auto.
      rew vec.
      apply ll_tps_lbang in Hx; tauto.

    + intros x (a & g & H2 & H3 & H4).
      apply IH1.
      apply proj2 in H3; subst; auto.
      rew vec; auto.
  
    + intros x (a & g & H2 & H3 & H4).
      apply IH1.
      exists a, g.
      repeat (split; auto).
      exists a, g.
      repeat (split; auto).
      apply proj2 in H3.
      subst; rew vec; auto.

    (* Cut Rule *)
 
    + intros x Hx.
      rewrite ll_tps_app in Hx.
      apply IH2.
      destruct Hx as (a & b & G1 & G2 & G3); subst.
      exists a, b; split; auto.
      split; auto.
      rewrite <- vec_zero_plus, vec_plus_comm.
      apply IH1; auto.

    (* â left *)

    + intros x Hx.
      apply IH1.
      destruct Hx as (c & g & ? & (a & b & ? & H2 & H3) & H4); subst.
      exists a, (vec_plus b g); split; auto.
      * rewrite vec_plus_assoc; trivial.
      * split; auto; exists b, g; auto.

    (* â right *)

    + intros x Hx.
      apply ll_tps_app in Hx.
      destruct Hx as (a & b & ? & H3 & H4); subst.
      exists a, b; split.
      * rewrite vec_plus_comm, vec_zero_plus; auto.
      * split; rewrite <- vec_zero_plus, vec_plus_comm.
        - apply IH1; auto.
        - apply IH2; auto.

    (* â left & â right 1 & 2 *)

    + intros x Hx.
      destruct Hx as (u & g & ? & [ G1 | G1 ] & G2); subst.
      * apply IH1; exists u, g; auto.
      * apply IH2; exists u, g; auto.
    + intros x Hx; left; apply IH1; auto.
    + intros x Hx; right; apply IH1; auto.

    (* â left, â right *)

    + intros ? (? & _ & _ & [] & _).    
    + intros x _; red; trivial.

    (* đ unit left & right *)

    + intros x (i & g & ? & H2 & H3); subst.
      red in H2; subst.
      rewrite vec_zero_plus.
      apply IH1; auto.
    + intros x Hx; red in Hx; subst.
      rewrite vec_zero_plus; red; trivial.
  Qed.

  (* This semantics is NOT complete for ILL 
     or even the (!,&,-o) fragment but it is
     complete for the EILL fragment *)
  
End trivial_phase_semantics.
