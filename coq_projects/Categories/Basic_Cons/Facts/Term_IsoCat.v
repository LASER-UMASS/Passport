From Categories Require Import Essentials.Notations.
From Categories Require Import Essentials.Types.
From Categories Require Import Essentials.Facts_Tactics.
From Categories Require Import Category.Main.
From Categories Require Import Functor.Functor Functor.Functor_Ops.
From Categories Require Import Basic_Cons.Terminal.
From Categories Require Import Cat.Cat Cat.Cat_Iso.

(** In this section we show that if a category C has a terminal object and D is
a category isomorphic to C, then D also has a terminal object. *)
Section Term_IsoCat.
  Context {C D : Category}
          (I : (C āā D ::> Cat)%isomorphism) (trm : (š_ C)%object).

  Program Definition Term_IsoCat : (š_ D)%object
    :=
      {|
        terminal := ((iso_morphism I) _o)%object trm;
        t_morph :=
          fun c =>
            match
              f_equal (fun w : (D āā» D)%functor => (w _o)%object c)
                      (right_inverse I)
              in _ = u return
              (u āā» _)%morphism
            with
              eq_refl => ((iso_morphism I) _a ((t_morph
                                                 trm ((Iā»Ā¹)%morphism _o c)))
                        )%morphism
            end;
        t_morph_unique :=
          fun c f g => _
      |}
  .

  Next Obligation.
  Proof.
    assert (H := f_equal
                   (fun w : (C āā» C)%functor => (w _o)%object (terminal trm))
                   (left_inverse I)).
    cbn in H.
    cut (
        match H in _ = u return
              (_ āā» u)%morphism
        with
        | eq_refl => ((I ā»Ā¹) _a f)%morphism
        end
        =
        match H in _ = u return
              (_ āā» u)%morphism
        with
        | eq_refl => ((I ā»Ā¹) _a g)%morphism
        end
    ).
    {
      intros H2.
      destruct H.
      match type of H2 with
        ?A = ?B =>
        assert (((iso_morphism I) _a A) = ((iso_morphism I) _a B))%morphism
          by (rewrite H2; trivial)
      end.
      rewrite <- (Cat_Iso_conv_inv_I_inv_I (Inverse_Isomorphism I) f).
      rewrite <- (Cat_Iso_conv_inv_I_inv_I (Inverse_Isomorphism I) g).
      apply f_equal.
      trivial.
    }
    {
      apply t_morph_unique.
    }
  Qed.

End Term_IsoCat.
