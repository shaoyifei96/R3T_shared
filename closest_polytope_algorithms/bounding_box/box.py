# -*- coding: utf-8 -*-
'''

@author: wualbert
'''
import numpy as np
from gurobipy import Model, GRB
from pypolycontain.lib.operations import to_AH_polytope
from pypolycontain.lib.objects import AH_polytope
from pypolycontain.lib.containment_encodings import constraints_AB_eq_CD

class AABB:
    def __init__(self, vertices, color=None, polytope=None):
        '''
        Creates an axis-aligned bounding bounding_box from two diagonal vertices
        :param vertices: a list of defining vertices with shape (2, dimensions)
        '''
        try:
            assert(len(vertices[0]) == len(vertices[1]))
        except AssertionError:
            print('Mismatched vertex dimensions')
            return
        self.dimension = len(vertices[0])
        self.u = np.asarray(vertices[0])
        self.v = np.asarray(vertices[1])
        #FIXME: use derived class
        self.polytope = polytope
        self.hash = None
        for d in range(self.dimension):
            if vertices[0][d]>vertices[1][d]:
                self.v[d], self.u[d] = vertices[0][d], vertices[1][d]
        #for visualizing
        if color is None:
            self.color=(np.random.random(),np.random.random(),np.random.random())
        else:
            self.color=color

    def __repr__(self):
        return "R^%d AABB with vertices "%(self.dimension) + np.array2string(self.u) +\
               ","+np.array2string(self.v)

    def __eq__(self, other):
        return (self.u==other.u).all()and\
               (self.v==other.v).all()and\
               (self.dimension==other.dimension)

    def __ne__(self, other):
        return not(self.__eq__(other))

    def __hash__(self):
        if self.hash is None:
            tpl = str((self.u, self.v, self.dimension))
            self.hash = hash(tpl)
        return self.hash

    def set_zonotope(self, zonotope):
        self.polytope = zonotope

    def overlaps(self, b2):
        '''
        U: lower corner. V: upper corner
        :param b2: bounding_box to compare to
        :return:
        '''
        u1_leq_v2 = np.less_equal(self.u,b2.v)
        u2_leq_v1 = np.less_equal(b2.u, self.v)
        return u1_leq_v2.all() and u2_leq_v1.all()

def AABB_centroid_edge(c, edge_lengths):
    '''
    Creates an AABB with centroid c and edge lengths
    :param centroid:
    :param edge_lengthes:
    :return:
    '''
    u = c-edge_lengths/2
    v = c+edge_lengths/2
    return AABB([np.ndarray.flatten(u),np.ndarray.flatten(v)])


def overlaps(a,b):
    return a.overlaps(b)

def point_in_box(point,box):
    for dim in range(box.dimension):
        if box.u[dim] > point[dim] or box.v[dim] < point[dim]:
            return False
    else:
        return True

def point_to_box_distance(point, box):
    out_range_dim = []
    for dim in range(box.dimension):
        if box.u[dim] < point[dim] and box.v[dim] > point[dim]:
            pass
        else:
            out_range_dim.append(min(abs(box.u[dim]-point[dim]), abs(box.v[dim]-point[dim])))
    return np.linalg.norm(out_range_dim)

def point_to_box_dmax(point, box):
    '''
    Computes the largest distance between point and x where x is in the box
    :param point:
    :param box:
    :return:
    '''
    farthest_point = np.zeros(box.dimension)
    for dim in range(box.dimension):
        if abs(box.u[dim]-point[dim])<abs(box.v[dim]-point[dim]):
            farthest_point[dim] = box.v[dim]
        else:
            farthest_point[dim] = box.u[dim]
    return np.linalg.norm(farthest_point-np.asarray(point))

def box_to_box_distance(query_box, box):
    out_range_dim = []
    for dim in range(box.dimension):
        if (box.u[dim] < query_box.u[dim] and box.v[dim] > query_box.u[dim]) or \
            (box.u[dim] < query_box.v[dim] and box.v[dim] > query_box.v[dim]) or \
            (query_box.u[dim] < box.u[dim] and query_box.v[dim] > box.u[dim]) or \
            (query_box.u[dim] < box.v[dim] and query_box.v[dim] > box.v[dim]):
            pass
        else:
            out_range_dim.append(min(min(abs(box.u[dim]-query_box.u[dim]), abs(box.v[dim]-query_box.u[dim])),
                                 min(abs(box.u[dim]-query_box.v[dim]), abs(box.v[dim]-query_box.v[dim]))))
    return np.linalg.norm(out_range_dim)

def zonotope_to_box(z, return_AABB = False):
    model = Model("zonotope_AABB")
    model.setParam('OutputFlag', False)
    dim=z.x.shape[0]
    p=np.empty((z.G.shape[1],1),dtype='object')
    #find extremum on each dimension
    results = [np.empty(z.x.shape[0]), np.empty(z.x.shape[0])]
    x = np.empty((z.x.shape[0],1),dtype='object')
    for row in range(p.shape[0]):
        p[row,0]=model.addVar(lb=-1,ub=1) #TODO: Generalize to AH polytopes. Use linear constraints on p.
        # See AH_polytope.py in pypolycontain.bounding_box: constraints_list_of_tuples(model,[(self.P.H,p),(-np.eye(self.P.h.shape[0]),self.P.h)],sign="<")
    model.update()
    for d in range(dim):
        x[d] = model.addVar(obj=0,lb=-GRB.INFINITY,ub=GRB.INFINITY)
    constraints_AB_eq_CD(model,np.eye(dim),x-z.x,z.G,p)

    for d in range(dim):
        x[d,0].Obj = 1
        #find minimum
        model.ModelSense = 1
        model.update()
        model.optimize()
        assert(model.Status==2)
        results[0][d] = x[d,0].X
        #find maximum
        model.ModelSense = -1
        model.update()
        model.optimize()
        assert(model.Status==2)
        results[1][d] = x[d,0].X
        #reset coefficient
        x[d,0].obj = 0
    if return_AABB:
        box = AABB(results, color=z.color, polytope=z)
        return box
    else:
        return np.ndarray.flatten(np.asarray(results))

def AH_polytope_to_box(ahp, return_AABB = False):
    # if ahp.type == 'zonotope':
    #     return zonotope_to_box(ahp, return_AABB=return_AABB)
    # if ahp.type != 'AH_polytope':
    #     print('Warning: Input is not AH-Polytope!')
    assert(isinstance(ahp, AH_polytope))
    ahp = to_AH_polytope(ahp)
    model = Model("ah_polytope_AABB")
    model.setParam('OutputFlag', False)
    dim=ahp.t.shape[0]
    #find extremum on each dimension
    lu = np.zeros([2, ahp.t.shape[0]], dtype='float')
    x = np.empty((ahp.P.H.shape[1], 1), dtype='object')
    #construct decision variables l and u
    model.update()
    #construct decision variable x
    for d in range(x.shape[0]):
        x[d] = model.addVar(obj=0,lb=-GRB.INFINITY,ub=GRB.INFINITY)

    #Force model update
    model.update()
    #add polytope constraint Hx<=h
    for d in range(ahp.P.h.shape[0]):
        model.addConstr(np.dot(ahp.P.H[d,:], x)[0] <= ahp.P.h[d])
    model.update()

    for d in range(dim):
        #find minimum
        model.setObjective((ahp.t+np.dot(ahp.T, x))[d,0], GRB.MINIMIZE)
        model.update()
        model.optimize()
        if model.Status!=2:
            print("WARNING: AH-polytope discarded")
            lu[0,:] = np.ndarray.flatten(ahp.t)
            lu[1,:] = np.ndarray.flatten(ahp.t)+10**-3
            return np.ndarray.flatten(lu)
#            print ahp.P.H,ahp.P.h
#            print ahp.T,ahp.t
#            return
        assert(model.Status==2)
        lu[0,d] = ahp.t[d,0]
        for i in range(x.shape[0]):
            lu[0,d]+=ahp.T[d,i]*x[i,0].X
        #find maximum
        model.setObjective((ahp.t+np.dot(ahp.T, x))[d,0], GRB.MAXIMIZE)
        model.update()
        model.optimize()
        assert(model.Status==2)
        lu[1,d] = ahp.t[d,0]
        for i in range(x.shape[0]):
            lu[1,d]+=ahp.T[d,i]*x[i,0].X

    if return_AABB:
        box = AABB(lu, polytope=ahp)
        return box
    else:
        return np.ndarray.flatten(lu)
