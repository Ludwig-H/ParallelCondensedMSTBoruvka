
# ParallelCondensedMSTBoruvka (fixed edges)

- MST Cython (robuste, mono-thread), retourne aussi les indices d'arêtes d'origine.
- Condensed tree exact façon HDBSCAN avec **multi-enfants** (arêtes MST de même rayon).
- **EOM exact** avec **poids** (venus de `Points`), stabilité = poids × (λ_death − λ_birth).
- **Splitting(loss)**: on retire les **ponts** (arêtes exactement au rayon de l'événement).
- **Correct edge accounting**: `sum(len(e) for e in Aretes) == N-1` sur graphe connexe.
  - `Aretes[0..M]` = uniquement les **arêtes internes** aux **feuilles** (pas les ponts de leur événement).
  - `Aretes[M+1+i]` = **ponts** de la ligne `Z[i]` (un lot par **ligne**, y compris si l'événement génère k−1 lignes).

Voir `notebooks/colab_demo.ipynb` pour un test end-to-end.
