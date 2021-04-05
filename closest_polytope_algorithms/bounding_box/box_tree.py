import numpy as np


class BoxNode:
    def __init__(self, box, key_dim=None, key_value=None, left_child=None, right_child=None, parent=None):
        self.left_child = left_child    #left child node
        self.right_child = right_child  #right child node
        self.parent = parent    #parent node
        self.box = box
        self.left_child_min_u = np.inf
        self.left_child_max_v = -np.inf
        self.right_child_min_u = np.inf
        self.right_child_max_v = -np.inf
        # TODO: find best axis for sorting
        self.key_dim = key_dim  #key dimension for sorting
        self.key_value = key_value #key for sorting
        self.update_for_children()

    def __repr__(self):
        if self.parent is None:
            parent_rep = 'None'
        else:
            parent_rep = self.parent.box.__repr__()

        if self.left_child is None:
            left_child_rep = 'None'
        else:
            left_child_rep = self.left_child.box.__repr__()

        if self.right_child is None:
            right_child_rep = 'None'
        else:
            right_child_rep = self.right_child.box.__repr__()

        return '\nBoxNode: ' + '\n' + self.box.__repr__() +'\n'+\
                '   parent bounding_box: ' + parent_rep +'\n'+\
                '   left child bounding_box: ' + left_child_rep +'\n'+\
                '   right child bounding_box: ' + right_child_rep +'\n'+\
                '   left child range '+ str((self.left_child_min_u, self.left_child_max_v))+ '\n'+\
                '   right child range ' + str((self.right_child_min_u, self.right_child_max_v))+ '\n'+\
                '   split by dim ' + str(self.key_dim) + ' at ' + str(self.key_value) + '\n'

    def update_for_children(self):
        if self.left_child:
            self.left_child_min_u = np.minimum(np.minimum(self.left_child.left_child_min_u, \
                                                          self.left_child.right_child_min_u,),\
                                                          self.left_child.box.u)
            self.left_child_max_v = np.maximum(np.maximum(self.left_child.left_child_max_v, \
                                                          self.left_child.right_child_max_v,),\
                                                          self.left_child.box.v)

        if self.right_child:
            self.right_child_min_u = np.minimum(np.minimum(self.right_child.left_child_min_u, \
                                                          self.right_child.right_child_min_u,),\
                                                          self.right_child.box.u)
            self.right_child_max_v = np.maximum(np.maximum(self.right_child.left_child_max_v, \
                                                          self.right_child.right_child_max_v,),\
                                                          self.right_child.box.v)
    def set_parent(self, parent):
        self.parent = parent

    def set_key_dim(self,key_dim):
        self.key_dim = key_dim

    def set_key_value(self,key_value):
        self.key_value = key_value

    def set_left_child(self, left_child):
        self.left_child = left_child
        self.update_for_children()

    def set_right_child(self, right_child):
        self.right_child = right_child
        self.update_for_children()

    def in_this_box(self,test_box):
        return self.box.overlaps(test_box)

    def evaluate_node(self, test_box, overlapping_box_list):
        '''

        :param test_box: box to be tested
        :param overlapping_box_list: stores overlapping AABBs upon completion of the function
        :return:
        '''
        if self.in_this_box(test_box):   #leaf branch
            overlapping_box_list.append(self.box)

        if self.left_child:     #exists a left child
            #check whether to evaluate left branch
            vi_geq_l_umin = np.invert(np.less(test_box.v, self.left_child_min_u))
            ui_leq_l_vmax = np.invert(np.greater(test_box.u, self.left_child_max_v))
            if vi_geq_l_umin.all() and ui_leq_l_vmax.all():
                self.left_child.evaluate_node(test_box,overlapping_box_list)

        if self.right_child:    #exists a right child
            if test_box.v[self.key_dim] >= self.key_value:
                #check whether to evaluate right branch
                vi_geq_r_umin = np.invert(np.less(test_box.v, self.right_child_min_u))
                ui_leq_r_vmax = np.invert(np.greater(test_box.u, self.right_child_max_v))
                if vi_geq_r_umin.all() and ui_leq_r_vmax.all():
                    self.right_child.evaluate_node(test_box,overlapping_box_list)

def find_median(box_nodes, q):
    '''
    Given a list of BoxNodes and the axis of interest q, find the median that splits the BoxNodes evenly
    :param box_nodes: a list of BoxNodes
    :param q: the dimension to perform binary splitting
    :return: the median of the box_nodes' index
    '''
    uqs = np.zeros([len(box_nodes),1])
    for i,bn in enumerate(box_nodes):
        uqs[i] = bn.box.u[q]
    return np.median(uqs)

def split_by_value(box_nodes, q, m):
    '''
    Given a list of BoxNodes, the dimension to perform binary splitting, and the splitting key, split the nodes
    :param box_nodes: list of BoxNodes
    :param q: the dimension to perform binary splitting
    :param m: split index
    :return: 2 tuple of the left and right BoxNode lists
    '''
    box_node_m = box_nodes[0]
    box_nodes_l = []
    box_nodes_r = []
    for i, bn in enumerate(box_nodes):
        if abs(bn.box.u[q]-m)<abs(box_node_m.box.u[q]-m):
            box_node_m = bn

    for i, bn in enumerate(box_nodes):
        if bn == box_node_m:
            continue
        if bn.box.u[q]<=m:
            box_nodes_l.append(bn)
        else:
            box_nodes_r.append(bn)

    return (box_node_m,box_nodes_l,box_nodes_r)

def binary_split(box_nodes_list, q=0, parent=None, recurse_q=None):
    '''
    Binary split a list of BoxNodes
    :param box_nodes_list: list of BoxNodes
    :param q: the dimension to perform binary splitting
    :param parent: the parent node of box_nodes. For preventing infinite recursions
    :return: node_m, the "root" node
    '''
    # Termination condition
    if len(box_nodes_list)==0:
        return None
    if len(box_nodes_list)==1:
        # at a leaf
        return box_nodes_list[0]
    q = np.mod(q, box_nodes_list[0].box.u.shape[0])
    m = find_median(box_nodes_list, q)
    box_node_m, box_nodes_l, box_nodes_r = split_by_value(box_nodes_list, q, m)
    if abs(len(box_nodes_l)-len(box_nodes_r))>1:    #unbalanced tree
        if recurse_q is None:
            return binary_split(box_nodes_list, q + 1, parent=parent, recurse_q=q)
        elif q == np.mod(recurse_q-1, box_nodes_list[0].box.u.shape[0]):
            #FIXME: Find a better way to break ties
            pass
        else:
            return binary_split(box_nodes_list, q + 1, parent=box_node_m, recurse_q=recurse_q)
    #set split key dimension and value
    box_node_m.set_key_dim(q)
    box_node_m.set_key_value(m)

    left_child_node = binary_split(box_nodes_l,q+1, parent=box_node_m)
    right_child_node = binary_split(box_nodes_r,q+1, parent=box_node_m)
    box_node_m.set_left_child(left_child_node)
    box_node_m.set_right_child(right_child_node)
    box_node_m.set_parent(parent)
    if left_child_node is not None:
        left_child_node.set_parent(box_node_m)
    if right_child_node is not None:
        right_child_node.set_parent(box_node_m)
    return box_node_m