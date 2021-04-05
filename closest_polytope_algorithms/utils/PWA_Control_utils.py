from closest_polytope_algorithms.bounding_box.polytope_tree import PolytopeTree

def polytree_to_zonotope_tree(polytree):
    zonotopes = []
    for s in polytree.states:
        zonotopes.append(s.p)
    # print(zonotopes)
    zt = PolytopeTree(zonotopes)
    return zt