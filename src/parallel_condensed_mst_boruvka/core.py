import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed
from ._boruvka import boruvka_mst

_EPS = 1e-12


def _to_arrays(U, V, W):
    U = np.asarray(U, dtype=np.int32)
    V = np.asarray(V, dtype=np.int32)
    W = np.asarray(W, dtype=np.float64)
    if U.shape != V.shape or V.shape != W.shape or U.ndim != 1:
        raise ValueError("U,V,W must be 1D arrays of equal length")
    if (U >= V).any() or (U < 0).any() or (V < 0).any():
        raise ValueError("Require 0<=U[i]<V[i]")
    if (W < 0).any():
        raise ValueError("W must be >= 0")
    return U, V, W


def _node_weights(N, Points):
    Wn = np.zeros(int(N), dtype=np.float64)
    for lst in Points:
        for node, p in lst:
            Wn[int(node)] += float(p)
    return Wn


def _build_event_tree(N, U_mst, V_mst, W_mst, Eidx_mst, node_w):
    """
    Construit l'arbre d'événements multi-enfants à partir du MST :
      - Chaque groupe d'arêtes de même rayon r fusionne des composantes courantes.
      - Pour chaque événement parent pid, on stocke:
         * comp_children[pid] : liste des enfants (composantes avant fusion)
         * r_of[pid] : rayon r
         * comp_nodes[pid] : ensemble des noeuds (0..N-1) de la composante
         * comp_weight[pid] : poids total (somme des poids des enfants)
         * event_edges[pid] : indices des arêtes du MST utilisées comme ponts à ce rayon
      - NB: on ne "répartit" pas encore les arêtes internes aux feuilles ici; on le fera dans _build_Z_Aretes.
    """
    N = int(N)
    parent = {}
    comp_of_node = {i: i for i in range(N)}
    comp_nodes = {i: set([i]) for i in range(N)}
    comp_weight = {i: float(node_w[i]) for i in range(N)}
    comp_children = defaultdict(list)
    r_of = {i: 0.0 for i in range(N)}
    events = []
    event_edges = {}

    order = np.argsort(W_mst, kind="mergesort")
    i = 0
    next_id = N
    m = len(order)
    while i < m:
        r = float(W_mst[order[i]])
        group = []
        j = i
        while j < m and abs(float(W_mst[order[j]]) - r) <= _EPS:
            group.append(int(order[j]))
            j += 1

        # Graphe d'adjacence sur les composantes courantes induit par ces arêtes
        adj = defaultdict(list)
        for k in group:
            u = int(U_mst[k])
            v = int(V_mst[k])
            e = int(Eidx_mst[k])  # indice d'arête d'origine
            cu = comp_of_node[u]
            cv = comp_of_node[v]
            if cu == cv:
                continue
            adj[cu].append((cv, e))
            adj[cv].append((cu, e))

        # Parcours des CC de ce graphe pour créer un événement par CC
        seen = set()
        for root in list(adj.keys()):
            if root in seen:
                continue
            stack = [root]
            seen.add(root)
            comps = [root]
            bridges = set()
            while stack:
                x = stack.pop()
                for y, eorig in adj.get(x, []):
                    bridges.add(eorig)
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
                        comps.append(y)

            # Création du parent
            pid = next_id
            next_id += 1
            r_of[pid] = r
            comp_children[pid] = []
            comp_nodes[pid] = set()
            comp_weight[pid] = 0.0
            for ch in comps:
                parent[ch] = pid
                comp_children[pid].append(ch)
                comp_nodes[pid] |= comp_nodes[ch]
                comp_weight[pid] += comp_weight[ch]
            event_edges[pid] = sorted(bridges)

            # Mise à jour des appartenances de noeuds
            for node in comp_nodes[pid]:
                comp_of_node[node] = pid

            events.append({"id": pid, "r": r, "children": list(comps)})

        i = j

    return (
        events,
        parent,
        comp_nodes,
        comp_weight,
        comp_children,
        r_of,
        event_edges,
    )


def _condense_eom(
    events,
    parent,
    comp_nodes,
    comp_weight,
    comp_children,
    r_of,
    min_cluster_size,
):
    """
    Restreint aux événements éligibles (poids >= min_cluster_size) et calcule la
    stabilité EOM exacte:
      stabilité(C) = poids(C) * (lambda_death(C) - lambda_birth(C)),  lambda = 1/max(r, eps)
    """
    ev_ids = [ev["id"] for ev in events]
    eligible = {cid for cid in ev_ids if comp_weight.get(cid, 0.0) >= float(min_cluster_size)}
    if not eligible:
        return set(), {}, {}, {}, {}, {}

    # Parent éligible le plus proche
    eligible_parent = {}
    for cid in eligible:
        p = parent.get(cid, None)
        while p is not None and p not in eligible:
            p = parent.get(p, None)
        eligible_parent[cid] = p

    # Enfants éligibles directs
    eligible_children = {cid: [] for cid in eligible}
    for cid in eligible:
        p = eligible_parent[cid]
        if p is not None:
            eligible_children[p].append(cid)

    # lambda des événements
    lam = {}
    for ev_id in ev_ids:
        r = r_of.get(ev_id, 0.0)
        lam[ev_id] = 1.0 / max(r, _EPS) if r > 0.0 else 1.0 / _EPS

    lam_birth = {}
    lam_death = {}
    stability = {}
    for cid in eligible:
        p = eligible_parent[cid]
        lam_birth[cid] = lam[p] if p is not None else 0.0
        chs = comp_children.get(cid, [])
        lam_children = [lam[ch] for ch in chs if ch in lam]
        lam_death[cid] = min(lam_children) if lam_children else lam[cid]
        stability[cid] = max(0.0, comp_weight[cid] * max(0.0, lam_death[cid] - lam_birth[cid]))

    return eligible, eligible_children, eligible_parent, stability, lam_birth, lam_death


def _descendants_events(ev_children, root):
    """
    Renvoie la liste des événements (ids >= N) dans le sous-arbre de 'root' (root inclus si c'est un événement).
    """
    out = []
    stack = [root]
    while stack:
        x = stack.pop()
        if x in ev_children:
            out.append(x)
            stack.extend(ev_children[x])
    return out


def _build_Z_Aretes(
    N,
    U,
    V,
    W,
    events,
    eligible,
    eligible_children,
    stability,
    comp_weight,
    r_of,
    event_edges,
):
    """
    Construit:
      - Z (linkage binaire) sur les nœuds éligibles (événements)
      - Aretes:
          * Aretes[0..M]   : arêtes internes de chaque FEUILLE (base clusters du condensed tree)
          * Aretes[M+1+i]  : arêtes de pont (toutes) de l'ÉVÉNEMENT correspondant à la ligne Z[i]
    Règle de distribution des arêtes du MST:
      - Chaque arête du MST appartient à un unique 'event' (celui où elle apparaît).
      - Si l'event est "éligible interne" (a des enfants éligibles), ses arêtes vont dans Aretes[M+1+i] (lignes Z).
      - Sinon (event non éligible, ou event éligible FEUILLE), ses arêtes remontent dans la feuille éligible ancêtre (base cluster).
    """
    # Dictionnaire enfants pour tous les événements
    ev_children = {ev["id"]: list(ev["children"]) for ev in events}

    # Partition éligibles: feuilles et internes
    leaves_eligible = [cid for cid in eligible if len(eligible_children.get(cid, [])) == 0]
    internal_eligible = [cid for cid in eligible if len(eligible_children.get(cid, [])) > 0]
    leaves_eligible.sort(key=lambda x: (r_of.get(x, 0.0), x))
    internal_eligible.sort(key=lambda x: (r_of.get(x, 0.0), x))
    leaf_index = {cid: i for i, cid in enumerate(leaves_eligible)}

    # Ensemble des events "avec ligne Z" (internes éligibles)
    eligible_internal_set = set(internal_eligible)

    # Arêtes internes des feuilles: union des event_edges de tout leur sous-arbre,
    # en EXCLUANT les events internes éligibles (ces events ont leurs arêtes dans les lignes Z)
    Aretes = []
    for leaf in leaves_eligible:
        # Tous les events (>= N) dans le sous-arbre du leaf (incluant le leaf s'il est event)
        sub_events = _descendants_events(ev_children, leaf)
        leaf_edges = set()
        for e in sub_events:
            if e not in eligible_internal_set:
                leaf_edges.update(event_edges.get(e, []))
        # Pour un leaf, si lui-même n'a pas d'enfants éligibles, on INCLUT ses event_edges
        # (ils ne seront pas dans une ligne Z)
        if leaf not in eligible_internal_set:
            leaf_edges.update(event_edges.get(leaf, []))
        Aretes.append(sorted(leaf_edges))

    # Construction Z (chaînage binaire) + arêtes de pont par ligne
    rows = []
    per_row_bridges = []
    comp_to_row_id = dict(leaf_index)
    current_id = len(leaves_eligible)

    for p in internal_eligible:
        chs = sorted(eligible_children[p], key=lambda x: (r_of.get(x, 0.0), x))
        ids = []
        for ch in chs:
            if ch not in comp_to_row_id:
                # Garantit un id pour l'enfant si jamais
                comp_to_row_id[ch] = current_id
                # Son paquet d'arêtes internes est (re)calculé ci-dessus pour les feuilles;
                # si par construction un enfant interne apparaît ici, il deviendra parent plus tard.
                Aretes.append([])  # placeholder, n'affecte pas les feuilles
                current_id += 1
            ids.append(comp_to_row_id[ch])

        r = r_of[p]
        # On chaîne en binaire: chaque ligne représente une fusion à rayon r
        while len(ids) > 1:
            a = ids.pop(0)
            b = ids.pop(0)
            rows.append([float(a), float(b), float(r), float(comp_weight[p]), float(stability.get(p, 0.0))])
            # Pour le splitting correct à ce niveau, on DOIT fournir toutes les arêtes de pont de l'événement 'p'
            per_row_bridges.append(sorted(event_edges.get(p, [])))
            new_id = current_id
            current_id += 1
            ids.insert(0, new_id)

        # L'event 'p' est représenté par l'id courant (dernier new_id introduit)
        comp_to_row_id[p] = ids[0] if ids else comp_to_row_id.get(p, current_id)

    Z = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5), dtype=np.float64)
    M = Z.shape[0]
    # Ajoute les arêtes de pont par ligne (alignement 1:1 avec Z[i])
    Aretes.extend(per_row_bridges)

    return Z, Aretes


def _selected_ids_eom(Z):
    if Z.size == 0:
        return set()
    M = Z.shape[0]
    base = M + 1
    children = defaultdict(list)
    stab = {}
    for i in range(M):
        pid = base + i
        children[pid].extend([int(Z[i, 0]), int(Z[i, 1])])
        stab[pid] = float(Z[i, 4])

    def dfs(u):
        if u < base:
            return 0.0, {u}
        svals = 0.0
        parts = []
        for v in children[u]:
            val, sub = dfs(v)
            svals += val
            parts.append(sub)
        here = stab.get(u, 0.0)
        if here >= svals:
            return here, {u}
        out = set()
        for sub in parts:
            out |= sub
        return svals, out

    _, sel = dfs(base + M - 1)
    return sel


def _selected_ids_leaf(Z):
    if Z.size == 0:
        return set()
    return set(range(Z.shape[0] + 1))


def _selected_ids_threshold(Z, thr):
    if Z.size == 0:
        return set()
    M = Z.shape[0]
    base = M + 1
    parent = np.arange(base, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(M):
        if float(Z[i, 2]) <= thr:
            unite(int(Z[i, 0]), int(Z[i, 1]))
    comps = defaultdict(list)
    for i in range(M + 1):
        comps[find(i)].append(i)
    sel = set()
    for nodes in comps.values():
        sel.update(nodes)
    return sel


def ParallelCondensedMSTBoruvka(U, V, W, N, Points, min_cluster_size, N_CPU_dispos=0, verbose=False):
    # Supporte aussi l'ordre (N, U, V, W, ...)
    if np.isscalar(U) and not np.isscalar(V):
        N_scalar = int(U)
        U, V, W, N = V, W, N, N_scalar

    U, V, W = _to_arrays(U, V, W)
    N = int(N)
    node_w = _node_weights(N, Points)

    # MST (Cython)
    U_mst, V_mst, W_mst, Eidx_mst = boruvka_mst(U, V, W, N, 0)

    # Arbre d'événements (multi-enfants)
    events, parent, comp_nodes, comp_weight, comp_children, r_of, event_edges = _build_event_tree(
        N, U_mst, V_mst, W_mst, Eidx_mst, node_w
    )

    # Condensed tree + EOM exact
    eligible, eligible_children, eligible_parent, stability, lam_birth, lam_death = _condense_eom(
        events, parent, comp_nodes, comp_weight, comp_children, r_of, float(min_cluster_size)
    )
    if not eligible:
        return np.zeros((0, 5), dtype=np.float64), []

    # Z + Aretes
    Z, Aretes = _build_Z_Aretes(
        N,
        U,
        V,
        W,
        events,
        eligible,
        eligible_children,
        stability,
        comp_weight,
        r_of,
        event_edges,
    )
    return Z, Aretes


def GetClusters(Z, Aretes, U, V, method, Points, N_CPU_dispos=0, splitting=None, verbose=False):
    U = np.asarray(U, dtype=np.int32)
    V = np.asarray(V, dtype=np.int32)
    M = Z.shape[0]
    base = M + 1

    # Sélection des ids (feuilles / EOM / seuil)
    if isinstance(method, str):
        m = method.lower()
        if m == "eom":
            selected = _selected_ids_eom(Z)
        elif m == "leaf":
            selected = _selected_ids_leaf(Z)
        else:
            raise ValueError("method must be eom, leaf or float threshold")
    else:
        thr = float(method)
        if thr <= 0.0:
            raise ValueError("threshold must be >0")
        selected = _selected_ids_threshold(Z, thr)

    # Split optionnel (par loss) en retirant les ponts de la ligne (Aretes[M+1+i])
    if splitting is not None and selected:
        selected = _apply_recursive_splitting(
            Z,
            Aretes,
            selected,
            U,
            V,
            Points,
            splitting,
            int(N_CPU_dispos) if N_CPU_dispos else 0,
        )

    # Arêtes d'un cluster id (réunion récursive)
    children = defaultdict(list)
    for i in range(M):
        children[base + i].extend([int(Z[i, 0]), int(Z[i, 1])])

    def gather_edges(cid):
        if cid < base:
            return set(int(e) for e in Aretes[cid])
        idx = cid - base
        es = set(int(e) for e in Aretes[base + idx])
        for ch in children.get(cid, []):
            es |= gather_edges(ch)
        return es

    sel_sorted = sorted(selected)

    def node_set_for(cid):
        edges = gather_edges(cid)
        nodes = set()
        for e in edges:
            if 0 <= e < len(U):
                nodes.add(int(U[e]))
                nodes.add(int(V[e]))
        return cid, nodes

    if N_CPU_dispos and len(sel_sorted) > 1:
        pairs = Parallel(n_jobs=int(N_CPU_dispos))(delayed(node_set_for)(cid) for cid in sel_sorted)
    else:
        pairs = [node_set_for(cid) for cid in sel_sorted]
    node_sets = dict(pairs)

    # Labels pondérés pour Points
    labels = []
    for i, lst in enumerate(Points):
        acc = defaultdict(float)
        for node, wgt in lst:
            hit = False
            for cid in sel_sorted:
                if int(node) in node_sets[cid]:
                    acc[cid] += float(wgt)
                    hit = True
            if not hit:
                acc[-1] += float(wgt)
        if len(acc) == 1:
            (cid, wt), = acc.items()
            labels.append((int(cid), float(wt)))
        else:
            labels.append([(int(c), float(w)) for c, w in sorted(acc.items())])
    return labels


def _apply_recursive_splitting(Z, Aretes, selected, U, V, Points, loss_func, n_jobs=0):
    """
    Splitting récursif:
      - pour un cluster (ligne Z[i]), on retire Aretes[M+1+i] (toutes les arêtes de pont de l'événement)
      - on reconstruit les sous-composantes avec les arêtes internes des feuilles descendantes
      - si sum(loss(sous-clusters)) <= loss(parent), on splitte et on continue récursivement.
    """
    M = Z.shape[0]
    base = M + 1

    children = defaultdict(list)
    for i in range(M):
        children[base + i].extend([int(Z[i, 0]), int(Z[i, 1])])

    def leaves_of(cid):
        if cid < base:
            return [cid]
        L = []
        for ch in children.get(cid, []):
            L.extend(leaves_of(ch))
        return L

    def nodes_from_leafset(L):
        s = set()
        for lid in L:
            for e in Aretes[lid]:
                s.add(int(U[e]))
                s.add(int(V[e]))
        return s

    def pts_from_leafset(L):
        nodes = nodes_from_leafset(L)
        acc = defaultdict(float)
        for idx, lst in enumerate(Points):
            for n, w in lst:
                if int(n) in nodes:
                    acc[idx] += float(w)
        return [(i, w) for i, w in acc.items()]

    def groups_after_removing_bridges(cid):
        if cid < base:
            return []
        row = cid - base
        bridges = set(int(e) for e in Aretes[base + row])
        L = leaves_of(cid)
        nodes = set()
        es = []
        for lid in L:
            for e in Aretes[lid]:
                u = int(U[e])
                v = int(V[e])
                nodes.add(u)
                nodes.add(v)
                if e not in bridges:
                    es.append((u, v))
        if not nodes:
            return []
        parent = {u: u for u in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def unite(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for u, v in es:
            unite(u, v)

        comps = defaultdict(set)
        for u in nodes:
            comps[find(u)].add(u)

        groups = []
        for cn in comps.values():
            lids = []
            for lid in L:
                s = set()
                for e in Aretes[lid]:
                    s.add(int(U[e]))
                    s.add(int(V[e]))
                if s & cn:
                    lids.append(lid)
            if lids:
                groups.append(sorted(lids))
        return groups

    out = set(selected)
    stack = [cid for cid in list(selected) if cid >= base]
    while stack:
        cid = stack.pop()
        parent_loss = float(loss_func(pts_from_leafset(leaves_of(cid))))
        groups = groups_after_removing_bridges(cid)
        if not groups or len(groups) <= 1:
            continue

        def loss_g(g):
            return float(loss_func(pts_from_leafset(g)))

        if n_jobs and len(groups) > 1:
            losses = Parallel(n_jobs=n_jobs)(delayed(loss_g)(g) for g in groups)
        else:
            losses = [loss_g(g) for g in groups]

        if sum(losses) <= parent_loss + 1e-12:
            if cid in out:
                out.remove(cid)
            for g in groups:
                # on représente chaque sous-cluster par ses feuilles;
                # si certaines sont internes, elles seront ré-examinées au tour suivant
                for lid in g:
                    out.add(lid)
    return out
