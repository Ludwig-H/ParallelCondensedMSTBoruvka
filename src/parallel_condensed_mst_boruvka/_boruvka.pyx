# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free   # allocations C pures

ctypedef np.int32_t INT_t
ctypedef np.float64_t D_t

cdef inline int uf_find(int* parent, int x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

cdef inline void uf_union(int* parent, int* rank, int a, int b):
    a = uf_find(parent, a)
    b = uf_find(parent, b)
    if a == b:
        return
    if rank[a] < rank[b]:
        parent[a] = b
    elif rank[a] > rank[b]:
        parent[b] = a
    else:
        parent[b] = a
        rank[a] += 1

def boruvka_mst(np.ndarray[INT_t, ndim=1] U_arr,
                np.ndarray[INT_t, ndim=1] V_arr,
                np.ndarray[D_t,  ndim=1] W_arr,
                int N,
                int n_threads=0):
    """
    MST par passes à la Borůvka (mono-thread, robuste).
    Retourne (Uo, Vo, Wo, Ei) où Ei sont les indices des arêtes d'origine.
    """
    cdef Py_ssize_t m = U_arr.shape[0]
    cdef int n = N

    cdef const INT_t[:] U = U_arr
    cdef const INT_t[:] V = V_arr
    cdef const D_t[:]  W = W_arr

    # sorties de taille max n-1, qu'on retaillera à la fin
    cdef np.ndarray[INT_t, ndim=1] Uo_arr = np.empty(max(n - 1, 0), dtype=np.int32)
    cdef np.ndarray[INT_t, ndim=1] Vo_arr = np.empty(max(n - 1, 0), dtype=np.int32)
    cdef np.ndarray[D_t,  ndim=1] Wo_arr = np.empty(max(n - 1, 0), dtype=np.float64)
    cdef np.ndarray[INT_t, ndim=1] Ei_arr = np.empty(max(n - 1, 0), dtype=np.int32)
    cdef INT_t[:] Uo = Uo_arr
    cdef INT_t[:] Vo = Vo_arr
    cdef D_t[:]  Wo = Wo_arr
    cdef INT_t[:] Ei = Ei_arr

    if n <= 1 or m == 0:
        return Uo_arr, Vo_arr, Wo_arr, Ei_arr

    # DSU
    cdef int* parent = <int*> malloc(n * sizeof(int))
    cdef int* rank   = <int*> malloc(n * sizeof(int))
    if parent == NULL or rank == NULL:
        if parent != NULL: free(parent)
        if rank   != NULL: free(rank)
        raise MemoryError()

    cdef int i
    for i in range(n):
        parent[i] = i
        rank[i] = 0

    # meilleurs arcs sortants par composante
    cdef double* bestW = <double*> malloc(n * sizeof(double))
    cdef int*    bestU = <int*>    malloc(n * sizeof(int))
    cdef int*    bestV = <int*>    malloc(n * sizeof(int))
    cdef int*    bestE = <int*>    malloc(n * sizeof(int))
    if bestW == NULL or bestU == NULL or bestV == NULL or bestE == NULL:
        if bestW != NULL: free(bestW)
        if bestU != NULL: free(bestU)
        if bestV != NULL: free(bestV)
        if bestE != NULL: free(bestE)
        free(parent); free(rank)
        raise MemoryError()

    cdef double BIG = 1e300
    cdef double w
    cdef int edges_added = 0
    cdef int u, v, cu, cv, eidx, aidx
    cdef int updated

    # pas de with nogil ici : on manipule des ndarrays Python
    while edges_added < n - 1:
        # réinitialise les meilleurs arcs
        for i in range(n):
            bestW[i] = BIG
            bestU[i] = -1
            bestV[i] = -1
            bestE[i] = -1

        # 1) on cherche la meilleure arête sortante pour chaque composante
        for i in range(m):
            u = U[i]; v = V[i]; w = W[i]
            cu = uf_find(parent, u)
            cv = uf_find(parent, v)
            if cu == cv:
                continue
            if w < bestW[cu]:
                bestW[cu] = w
                bestU[cu] = u
                bestV[cu] = v
                bestE[cu] = i
            if w < bestW[cv]:
                bestW[cv] = w
                bestU[cv] = u
                bestV[cv] = v
                bestE[cv] = i

        # 2) on ajoute ces arêtes
        updated = 0
        for aidx in range(n):
            if bestU[aidx] == -1:
                continue
            u = bestU[aidx]
            v = bestV[aidx]
            w = bestW[aidx]
            eidx = bestE[aidx]
            cu = uf_find(parent, u)
            cv = uf_find(parent, v)
            if cu == cv:
                continue
            uf_union(parent, rank, cu, cv)
            Uo[edges_added] = u
            Vo[edges_added] = v
            Wo[edges_added] = w
            Ei[edges_added] = eidx
            edges_added += 1
            updated += 1
            if edges_added == n - 1:
                break

        if updated == 0:
            break

    # tronque les tableaux à edges_added
    Uo_arr = Uo_arr[:edges_added]
    Vo_arr = Vo_arr[:edges_added]
    Wo_arr = Wo_arr[:edges_added]
    Ei_arr = Ei_arr[:edges_added]

    free(bestW); free(bestU); free(bestV); free(bestE)
    free(parent); free(rank)
    return Uo_arr, Vo_arr, Wo_arr, Ei_arr
