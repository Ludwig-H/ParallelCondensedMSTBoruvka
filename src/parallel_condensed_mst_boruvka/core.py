import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed
from ._boruvka import boruvka_mst

_EPS = 1e-12

def _to_arrays(U,V,W):
    U = np.asarray(U, dtype=np.int32); V = np.asarray(V, dtype=np.int32); W = np.asarray(W, dtype=np.float64)
    if U.shape!=V.shape or V.shape!=W.shape or U.ndim!=1:
        raise ValueError('U,V,W must be 1D arrays of equal length')
    if (U>=V).any() or (U<0).any() or (V<0).any():
        raise ValueError('Require 0<=U[i]<V[i]')
    if (W<0).any():
        raise ValueError('W must be >= 0')
    return U,V,W

def _node_weights(N, Points):
    Wn = np.zeros(int(N), dtype=np.float64)
    for lst in Points:
        for node, p in lst:
            Wn[int(node)] += float(p)
    return Wn

def _build_event_tree(N, U_mst, V_mst, W_mst, Eidx_mst, node_w):
    N = int(N)
    parent = {}
    comp_of_node = {i:i for i in range(N)}
    comp_nodes = {i:set([i]) for i in range(N)}
    comp_weight = {i:float(node_w[i]) for i in range(N)}
    # edges_internal: intra-cluster edges (no bridges of the event itself)
    edges_internal = {i:[] for i in range(N)}
    # event edges (bridges at that radius)
    event_edges = {}
    comp_children = defaultdict(list)
    r_of = {i:0.0 for i in range(N)}
    events = []

    order = np.argsort(W_mst, kind='mergesort')
    i = 0; next_id = N; m = len(order)
    while i < m:
        r = float(W_mst[order[i]])
        group = []
        j = i
        while j < m and abs(float(W_mst[order[j]]) - r) <= _EPS:
            group.append(int(order[j])); j += 1
        # adjacency on current components induced by this group
        adj = defaultdict(list)
        for k in group:
            u = int(U_mst[k]); v = int(V_mst[k]); e = int(Eidx_mst[k])
            cu = comp_of_node[u]; cv = comp_of_node[v]
            if cu == cv: continue
            adj[cu].append((cv,e)); adj[cv].append((cu,e))
        seen = set()
        for root in list(adj.keys()):
            if root in seen: continue
            stack=[root]; seen.add(root)
            comps=[root]; bridges=set()
            while stack:
                x=stack.pop()
                for y,e in adj.get(x,[]):
                    bridges.add(e)
                    if y not in seen: seen.add(y); stack.append(y); comps.append(y)
            pid = next_id; next_id += 1
            r_of[pid] = r
            comp_nodes[pid] = set()
            comp_weight[pid] = 0.0
            # internal edges of parent = union of internal edges of children (no addition of bridges)
            edges_internal[pid] = []
            for ch in comps:
                parent[ch] = pid
                comp_children[pid].append(ch)
                comp_nodes[pid] |= comp_nodes[ch]
                comp_weight[pid] += comp_weight[ch]
                edges_internal[pid].extend(edges_internal[ch])
            event_edges[pid] = sorted(bridges)
            # update comp_of_node
            for node in comp_nodes[pid]:
                comp_of_node[node] = pid
            events.append({'id':pid,'r':r,'children':list(comps)})
        i = j
    return events, parent, comp_nodes, comp_weight, comp_children, r_of, edges_internal, event_edges

def _condense_eom(events, parent, comp_nodes, comp_weight, comp_children, r_of, min_cluster_size):
    ev_ids = [ev['id'] for ev in events]
    eligible = {cid for cid in ev_ids if comp_weight.get(cid,0.0) >= float(min_cluster_size)}
    if not eligible:
        return set(), {}, {}, {}, {}, {}
    # nearest eligible parent
    eligible_parent = {}
    for cid in eligible:
        p = parent.get(cid, None)
        while p is not None and p not in eligible:
            p = parent.get(p, None)
        eligible_parent[cid] = p
    eligible_children = {cid:[] for cid in eligible}
    for cid in eligible:
        p = eligible_parent[cid]
        if p is not None: eligible_children[p].append(cid)
    lam = {cid:(1.0/max(r_of[cid], _EPS) if r_of[cid]>0 else 1.0/_EPS) for cid in ev_ids}
    lam_birth = {}; lam_death = {}; stability = {}
    for cid in eligible:
        p = eligible_parent[cid]
        lam_birth[cid] = lam[p] if p is not None else 0.0
        chs = comp_children.get(cid, [])
        lams = [lam[ch] for ch in chs if ch in lam]
        lam_death[cid] = min(lams) if lams else lam[cid]
        stability[cid] = max(0.0, comp_weight[cid] * max(0.0, lam_death[cid] - lam_birth[cid]))
    return eligible, eligible_children, eligible_parent, stability, lam_birth, lam_death

def _build_Z_Aretes(N, U, V, W, events, eligible, eligible_children, stability, comp_weight, r_of, edges_internal, event_edges):
    # leaves (eligible with no eligible child)
    leaves = [cid for cid in eligible if len(eligible_children.get(cid,[]))==0]
    leaves.sort(key=lambda x: (r_of.get(x,0.0), x))
    leaf_index = {cid:i for i,cid in enumerate(leaves)}
    # Aretes for leaves: ONLY internal edges (no bridges of their own event)
    Aretes = [sorted(set(edges_internal.get(cid, []))) for cid in leaves]
    rows = []; per_row_bridges = []
    comp_to_row_id = dict(leaf_index)
    current_id = len(leaves)
    internal_nodes = [cid for cid in eligible if len(eligible_children.get(cid,[]))>0]
    internal_nodes.sort(key=lambda x: (r_of.get(x,0.0), x))
    for p in internal_nodes:
        chs = sorted(eligible_children[p], key=lambda x: (r_of.get(x,0.0), x))
        ids = []
        for ch in chs:
            if ch not in comp_to_row_id:
                comp_to_row_id[ch] = current_id
                Aretes.append(sorted(set(edges_internal.get(ch, []))))
                current_id += 1
            ids.append(comp_to_row_id[ch])
        r = r_of[p]
        while len(ids) > 1:
            a = ids.pop(0); b = ids.pop(0)
            rows.append([float(a), float(b), float(r), float(comp_weight[p]), float(stability.get(p,0.0))])
            per_row_bridges.append(sorted(set(event_edges.get(p, []))))  # attach bridges to THIS row
            new_id = current_id; current_id += 1
            ids.insert(0, new_id)
        comp_to_row_id[p] = ids[0] if ids else comp_to_row_id.get(p, current_id)
    Z = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0,5), dtype=np.float64)
    M = Z.shape[0]
    # Append bridges for each row, aligned 1:1
    Aretes.extend(per_row_bridges)
    return Z, Aretes

def _selected_ids_eom(Z):
    if Z.size==0: return set()
    M = Z.shape[0]; base = M+1
    children = defaultdict(list); stab = {}
    for i in range(M):
        pid = base+i; children[pid].extend([int(Z[i,0]), int(Z[i,1])]); stab[pid]=float(Z[i,4])
    def dfs(u):
        if u < base: return 0.0, {u}
        svals=0.0; sel=set()
        parts=[]
        for v in children[u]:
            val, sub = dfs(v); svals += val; parts.append(sub)
        here = stab.get(u,0.0)
        if here >= svals: return here, {u}
        out=set();
        for sub in parts: out |= sub
        return svals, out
    _, sel = dfs(base+M-1)
    return sel

def _selected_ids_leaf(Z):
    if Z.size==0: return set()
    return set(range(Z.shape[0]+1))

def _selected_ids_threshold(Z, thr):
    if Z.size==0: return set()
    M = Z.shape[0]; base = M+1
    parent = np.arange(base, dtype=np.int32)
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def unite(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    for i in range(M):
        if float(Z[i,2]) <= thr: unite(int(Z[i,0]), int(Z[i,1]))
    comps=defaultdict(list)
    for i in range(M+1): comps[find(i)].append(i)
    sel=set()
    for nodes in comps.values(): sel.update(nodes)
    return sel

def ParallelCondensedMSTBoruvka(U,V,W,N,Points,min_cluster_size,N_CPU_dispos=0,verbose=False):
    if np.isscalar(U) and not np.isscalar(V):
        N_scalar=int(U); U,V,W,N = V,W,N,N_scalar
    U,V,W = _to_arrays(U,V,W); N=int(N)
    node_w = _node_weights(N, Points)
    U_mst, V_mst, W_mst, Eidx_mst = boruvka_mst(U,V,W,N,0)
    events, parent, comp_nodes, comp_weight, comp_children, r_of, edges_internal, event_edges = _build_event_tree(N,U_mst,V_mst,W_mst,Eidx_mst,node_w)
    eligible, eligible_children, eligible_parent, stability, lam_birth, lam_death = _condense_eom(events,parent,comp_nodes,comp_weight,comp_children,r_of,float(min_cluster_size))
    if not eligible:
        return np.zeros((0,5),dtype=np.float64), []
    Z, Aretes = _build_Z_Aretes(N,U,V,W,events,eligible,eligible_children,stability,comp_weight,r_of,edges_internal,event_edges)
    return Z, Aretes

def GetClusters(Z, Aretes, U, V, method, Points, N_CPU_dispos=0, splitting=None, verbose=False):
    U = np.asarray(U, dtype=np.int32); V=np.asarray(V, dtype=np.int32)
    M = Z.shape[0]; base = M+1
    if isinstance(method,str):
        m=method.lower()
        if m=='eom': selected=_selected_ids_eom(Z)
        elif m=='leaf': selected=_selected_ids_leaf(Z)
        else: raise ValueError('method must be eom, leaf or float threshold')
    else:
        thr=float(method);
        if thr<=0.0: raise ValueError('threshold must be >0')
        selected=_selected_ids_threshold(Z,thr)
    # Optional splitting (loss-based) uses the bridges at row level
    if splitting is not None and selected:
        selected = _apply_recursive_splitting(Z, Aretes, selected, U, V, Points, splitting, int(N_CPU_dispos) if N_CPU_dispos else 0)
    # Build child map
    children=defaultdict(list)
    for i in range(M): children[base+i].extend([int(Z[i,0]), int(Z[i,1])])
    def gather_edges(cid):
        if cid < base: return set(int(e) for e in Aretes[cid])
        idx = cid - base
        es = set(int(e) for e in Aretes[base+idx])
        for ch in children.get(cid,[]): es |= gather_edges(ch)
        return es
    sel_sorted = sorted(selected)
    def node_set_for(cid):
        edges = gather_edges(cid)
        nodes=set()
        for e in edges:
            if 0<=e<len(U): nodes.add(int(U[e])); nodes.add(int(V[e]))
        return cid, nodes
    pairs = [node_set_for(cid) for cid in sel_sorted] if not N_CPU_dispos else Parallel(n_jobs=int(N_CPU_dispos))(delayed(node_set_for)(cid) for cid in sel_sorted)
    node_sets = dict(pairs)
    labels = []
    for i,lst in enumerate(Points):
        acc=defaultdict(float)
        for node,wgt in lst:
            hit=False
            for cid in sel_sorted:
                if int(node) in node_sets[cid]: acc[cid]+=float(wgt); hit=True
            if not hit: acc[-1]+=float(wgt)
        labels.append((int(next(iter(acc))), float(next(iter(acc.values())))) if len(acc)==1 else [(int(c), float(w)) for c,w in sorted(acc.items())])
    return labels

def _apply_recursive_splitting(Z, Aretes, selected, U, V, Points, loss_func, n_jobs=0):
    M = Z.shape[0]; base = M+1
    children=defaultdict(list)
    for i in range(M): children[base+i].extend([int(Z[i,0]), int(Z[i,1])])
    def leaves_of(cid):
        if cid < base: return [cid]
        L=[]
        for ch in children.get(cid,[]): L.extend(leaves_of(ch))
        return L
    def nodes_from_leafset(L):
        s=set()
        for lid in L:
            for e in Aretes[lid]: s.add(int(U[e])); s.add(int(V[e]))
        return s
    def pts_from_leafset(L):
        nodes = nodes_from_leafset(L)
        acc=defaultdict(float)
        for idx,lst in enumerate(Points):
            for n,w in lst:
                if int(n) in nodes: acc[idx]+=float(w)
        return [(i,w) for i,w in acc.items()]
    def groups_after_removing_bridges(cid):
        if cid < base: return []
        row = cid - base
        bridges = set(int(e) for e in Aretes[base+row])
        L = leaves_of(cid)
        nodes=set(); es=[]
        for lid in L:
            for e in Aretes[lid]:
                u=int(U[e]); v=int(V[e]); nodes.add(u); nodes.add(v)
                if e not in bridges: es.append((u,v))
        if not nodes: return []
        parent={u:u for u in nodes}
        def find(x):
            while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
            return x
        def unite(a,b):
            ra,rb=find(a),find(b)
            if ra!=rb: parent[rb]=ra
        for u,v in es: unite(u,v)
        comps=defaultdict(set)
        for u in nodes: comps[find(u)].add(u)
        groups=[]
        for cn in comps.values():
            lids=[]
            for lid in L:
                s=set()
                for e in Aretes[lid]: s.add(int(U[e])); s.add(int(V[e]))
                if s & cn: lids.append(lid)
            if lids: groups.append(sorted(lids))
        return groups
    out=set(selected)
    stack=[cid for cid in list(selected) if cid>=base]
    while stack:
        cid = stack.pop()
        parent_loss = float(loss_func(pts_from_leafset(leaves_of(cid))))
        groups = groups_after_removing_bridges(cid)
        if not groups or len(groups)<=1: continue
        def loss_g(g): return float(loss_func(pts_from_leafset(g)))
        losses = Parallel(n_jobs=n_jobs)(delayed(loss_g)(g) for g in groups) if n_jobs and len(groups)>1 else [loss_g(g) for g in groups]
        if sum(losses) <= parent_loss + 1e-12:
            if cid in out: out.remove(cid)
            for g in groups:
                # represent each group by its leaf ids; in selected set on Z-ids, we fallback to leaves
                for lid in g: out.add(lid)
    return out