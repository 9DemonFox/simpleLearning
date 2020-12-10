import json
import uuid
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')


# 工具方法_2 treelib
class NodePropertyError(Exception):
    """Basic Node attribute error"""
    pass


class NodeIDAbsentError(NodePropertyError):
    """Exception throwed if a node's identifier is unknown"""
    pass


class NodePropertyAbsentError(NodePropertyError):
    """Exception throwed if a node's data property is not specified"""
    pass


class MultipleRootError(Exception):
    """Exception throwed if more than one root exists in a tree."""
    pass


class DuplicatedNodeIdError(Exception):
    """Exception throwed if an identifier already exists in a tree."""
    pass


class LinkPastRootNodeError(Exception):
    """
    Exception throwed in Tree.link_past_node() if one attempts
    to "link past" the root node of a tree.
    """
    pass


class InvalidLevelNumber(Exception):
    pass


class LoopError(Exception):
    """
    Exception thrown if trying to move node B to node A's position
    while A is B's ancestor.
    """
    pass


class Node(object):
    """
    Nodes are elementary objects that are stored in the `_nodes` dictionary of a Tree.
    Use `data` attribute to store node-specific data.
    """

    #: Mode constants for routine `update_fpointer()`.
    (ADD, DELETE, INSERT, REPLACE) = list(range(4))

    def __init__(self, tag=None, identifier=None, expanded=True, data=None):
        """Create a new Node object to be placed inside a Tree object"""

        #: if given as a parameter, must be unique
        self._identifier = None
        self._set_identifier(identifier)

        #: None or something else
        #: if None, self._identifier will be set to the identifier's value.
        if tag is None:
            self._tag = self._identifier
        else:
            self._tag = tag

        #: boolean
        self.expanded = expanded

        #: identifier of the parent's node :
        self._bpointer = None
        #: identifier(s) of the soons' node(s) :
        self._fpointer = list()

        #: User payload associated with this node.
        self.data = data

    def __lt__(self, other):
        return self.tag < other.tag

    def _set_identifier(self, nid):
        """Initialize self._set_identifier"""
        if nid is None:
            self._identifier = str(uuid.uuid1())
        else:
            self._identifier = nid

    @property
    def bpointer(self):
        """
        The parent ID of a node. This attribute can be
        accessed and modified with ``.`` and ``=`` operator respectively.
        """
        return self._bpointer

    @bpointer.setter
    def bpointer(self, nid):
        """Set the value of `_bpointer`."""
        if nid is not None:
            self._bpointer = nid
        else:
            # print("WARNING: the bpointer of node %s " \
            #      "is set to None" % self._identifier)
            self._bpointer = None

    @property
    def fpointer(self):
        """
        With a getting operator, a list of IDs of node's children is obtained. With
        a setting operator, the value can be list, set, or dict. For list or set,
        it is converted to a list type by the package; for dict, the keys are
        treated as the node IDs.
        """
        return self._fpointer

    @fpointer.setter
    def fpointer(self, value):
        """Set the value of `_fpointer`."""
        if value is None:
            self._fpointer = list()
        elif isinstance(value, list):
            self._fpointer = value
        elif isinstance(value, dict):
            self._fpointer = list(value.keys())
        elif isinstance(value, set):
            self._fpointer = list(value)
        else:  # TODO: add deprecated routine
            pass

    @property
    def identifier(self):
        """
        The unique ID of a node within the scope of a tree. This attribute can be
        accessed and modified with ``.`` and ``=`` operator respectively.
        """
        return self._identifier

    @identifier.setter
    def identifier(self, value):
        """Set the value of `_identifier`."""
        if value is None:
            print("WARNING: node ID can not be None")
        else:
            self._set_identifier(value)

    def is_leaf(self):
        """Return true if current node has no children."""
        if len(self.fpointer) == 0:
            return True
        else:
            return False

    def is_root(self):
        """Return true if self has no parent, i.e. as root."""
        return self._bpointer is None

    @property
    def tag(self):
        """
        The readable node name for human. This attribute can be accessed and
        modified with ``.`` and ``=`` operator respectively.
        """
        return self._tag

    @tag.setter
    def tag(self, value):
        """Set the value of `_tag`."""
        self._tag = value if value is not None else None

    def update_bpointer(self, nid):
        """Set the parent (indicated by the ``nid`` parameter) of a node."""
        self.bpointer = nid

    def update_fpointer(self, nid, mode=ADD, replace=None):
        """
        Update the children list with different modes: addition (Node.ADD or
        Node.INSERT) and deletion (Node.DELETE).
        """
        if nid is None:
            return

        if mode is self.ADD:
            self._fpointer.append(nid)

        elif mode is self.DELETE:
            if nid in self._fpointer:
                self._fpointer.remove(nid)

        elif mode is self.INSERT:  # deprecate to ADD mode
            print("WARNING: INSERT is deprecated to ADD mode")
            self.update_fpointer(nid)

        elif mode is self.REPLACE:
            if replace is None:
                raise NodePropertyError(
                    'Argument "repalce" should be provided when mode is {}'.format(mode)
                )

            ind = self._fpointer.index(nid)
            self._fpointer[ind] = replace

    def __repr__(self):
        name = self.__class__.__name__
        kwargs = [
            "tag={0}".format(self.tag),
            "identifier={0}".format(self.identifier),
            "data={0}".format(self.data),
        ]
        return "%s(%s)" % (name, ", ".join(kwargs))


class Tree(object):
    """Tree objects are made of Node(s) stored in _nodes dictionary."""

    #: ROOT, DEPTH, WIDTH, ZIGZAG constants :
    (ROOT, DEPTH, WIDTH, ZIGZAG) = list(range(4))
    node_class = Node

    def __contains__(self, identifier):
        """Return a list of the nodes'identifiers matching the
        identifier argument.
        """
        return [node for node in self._nodes
                if node == identifier]

    def __init__(self, tree=None, deep=False, node_class=None):
        """Initiate a new tree or copy another tree with a shallow or
        deep copy.
        """

        if node_class:
            assert issubclass(node_class, Node)
            self.node_class = node_class

        #: dictionary, identifier: Node object
        self._nodes = {}

        #: Get or set the identifier of the root. This attribute can be accessed and modified
        #: with ``.`` and ``=`` operator respectively.
        self.root = None

        if tree is not None:
            self.root = tree.root

            if deep:
                for nid in tree._nodes:
                    self._nodes[nid] = deepcopy(tree._nodes[nid])
            else:
                self._nodes = tree._nodes

    def __getitem__(self, key):
        """Return _nodes[key]"""
        try:
            return self._nodes[key]
        except KeyError:
            raise NodeIDAbsentError("Node '%s' is not in the tree" % key)

    def __len__(self):
        """Return len(_nodes)"""
        return len(self._nodes)

    def __setitem__(self, key, item):
        """Set _nodes[key]"""
        self._nodes.update({key: item})

    def __str__(self):
        self._reader = ""

        def write(line):
            self._reader += line.decode('utf-8') + "\n"

        self.__print_backend(func=write)
        return self._reader

    def __print_backend(self, nid=None, level=ROOT, idhidden=True, filter=None,
                        key=None, reverse=False, line_type='ascii-ex',
                        data_property=None, func=print):

        # Factory for proper get_label() function
        def get_label(node):
            # return "%s[%s]" % (getattr(node.data, data_property), node.identifier)
            return "%s[%s]" % (node.tag, node.data)

        # legacy ordering
        if key is None:
            def key(node):
                return node

        # iter with func
        for pre, node in self.__get(nid, level, filter, key, reverse,
                                    line_type):
            label = get_label(node)
            func('{0}{1}'.format(pre, label).encode('utf-8'))

    def __get(self, nid, level, filter_, key, reverse, line_type):
        # default filter
        if filter_ is None:
            def filter_(node):
                return True

        # render characters
        dt = {
            'ascii': ('|', '|-- ', '+-- '),
            'ascii-ex': ('\u2502', '\u251c\u2500\u2500 ', '\u2514\u2500\u2500 '),
            'ascii-exr': ('\u2502', '\u251c\u2500\u2500 ', '\u2570\u2500\u2500 '),
            'ascii-em': ('\u2551', '\u2560\u2550\u2550 ', '\u255a\u2550\u2550 '),
            'ascii-emv': ('\u2551', '\u255f\u2500\u2500 ', '\u2559\u2500\u2500 '),
            'ascii-emh': ('\u2502', '\u255e\u2550\u2550 ', '\u2558\u2550\u2550 '),
        }[line_type]

        return self.__get_iter(nid, level, filter_, key, reverse, dt, [])

    def __get_iter(self, nid, level, filter_, key, reverse, dt, is_last):
        dt_vline, dt_line_box, dt_line_cor = dt
        leading = ''
        lasting = dt_line_box

        nid = self.root if (nid is None) else nid
        if not self.contains(nid):
            raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)

        node = self[nid]

        if level == self.ROOT:
            yield "", node
        else:
            leading = ''.join(map(lambda x: dt_vline + ' ' * 3
            if not x else ' ' * 4, is_last[0:-1]))
            lasting = dt_line_cor if is_last[-1] else dt_line_box
            yield leading + lasting, node

        if filter_(node) and node.expanded:
            children = [self[i] for i in node.fpointer if filter_(self[i])]
            idxlast = len(children) - 1
            if key:
                children.sort(key=key, reverse=reverse)
            elif reverse:
                children = reversed(children)
            level += 1
            for idx, child in enumerate(children):
                is_last.append(idx == idxlast)
                for item in self.__get_iter(child.identifier, level, filter_,
                                            key, reverse, dt, is_last):
                    yield item
                is_last.pop()

    def __update_bpointer(self, nid, parent_id):
        """set self[nid].bpointer"""
        self[nid].update_bpointer(parent_id)

    def __update_fpointer(self, nid, child_id, mode):
        if nid is None:
            return
        else:
            self[nid].update_fpointer(child_id, mode)

    def __real_true(self, p):
        return True

    def add_node(self, node, parent=None):
        """
        Add a new node object to the tree and make the parent as the root by default.

        The 'node' parameter refers to an instance of Class::Node.
        """
        if not isinstance(node, self.node_class):
            raise OSError(
                "First parameter must be object of {}".format(self.node_class))

        if node.identifier in self._nodes:
            raise DuplicatedNodeIdError("Can't create node "
                                        "with ID '%s'" % node.identifier)

        pid = parent.identifier if isinstance(
            parent, self.node_class) else parent

        if pid is None:
            if self.root is not None:
                raise MultipleRootError("A tree takes one root merely.")
            else:
                self.root = node.identifier
        elif not self.contains(pid):
            raise NodeIDAbsentError("Parent node '%s' "
                                    "is not in the tree" % pid)

        self._nodes.update({node.identifier: node})
        self.__update_fpointer(pid, node.identifier, self.node_class.ADD)
        self.__update_bpointer(node.identifier, pid)

    def all_nodes(self):
        """Return all nodes in a list"""
        return list(self._nodes.values())

    def all_nodes_itr(self):
        """
        Returns all nodes in an iterator.
        Added by William Rusnack
        """
        return self._nodes.values()

    def children(self, nid):

        return [self[i] for i in self.is_branch(nid)]

    def contains(self, nid):
        return True if nid in self._nodes else False

    def create_node(self, tag=None, identifier=None, parent=None, data=None):

        node = self.node_class(tag=tag, identifier=identifier, data=data)
        self.add_node(node, parent)
        return node

    def depth(self, node=None):

        ret = 0
        if node is None:
            # Get maximum level of this tree
            leaves = self.leaves()
            for leave in leaves:
                level = self.level(leave.identifier)
                ret = level if level >= ret else ret
        else:
            # Get level of the given node
            if not isinstance(node, self.node_class):
                nid = node
            else:
                nid = node.identifier
            if not self.contains(nid):
                raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)
            ret = self.level(nid)
        return ret

    def filter_nodes(self, func):

        return filter(func, self.all_nodes_itr())

    def get_node(self, nid):

        if nid is None or not self.contains(nid):
            return None
        return self._nodes[nid]

    def is_branch(self, nid):
        """
        Return the children (ID) list of nid.
        Empty list is returned if nid does not exist
        """
        if nid is None:
            raise OSError("First parameter can't be None")
        if not self.contains(nid):
            raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)

        try:
            fpointer = self[nid].fpointer
        except KeyError:
            fpointer = []
        return fpointer

    def leaves(self, nid=None):
        """Get leaves of the whole tree or a subtree."""
        leaves = []
        if nid is None:
            for node in self._nodes.values():
                if node.is_leaf():
                    leaves.append(node)
        else:
            for node in self.expand_tree(nid):
                if self[node].is_leaf():
                    leaves.append(self[node])
        return leaves

    def level(self, nid, filter=None):

        return len([n for n in self.rsearch(nid, filter)]) - 1

    def move_node(self, source, destination):
        """
        Move node @source from its parent to another parent @destination.
        """
        if not self.contains(source) or not self.contains(destination):
            raise NodeIDAbsentError
        elif self.is_ancestor(source, destination):
            raise LoopError

        parent = self[source].bpointer
        self.__update_fpointer(parent, source, self.node_class.DELETE)
        self.__update_fpointer(destination, source, self.node_class.ADD)
        self.__update_bpointer(source, destination)

    def is_ancestor(self, ancestor, grandchild):
        """
        Check if the @ancestor the preceding nodes of @grandchild.

        :param ancestor: the node identifier
        :param grandchild: the node identifier
        :return: True or False
        """
        parent = self[grandchild].bpointer
        child = grandchild
        while parent is not None:
            if parent == ancestor:
                return True
            else:
                child = self[child].bpointer
                parent = self[child].bpointer
        return False

    @property
    def nodes(self):
        """Return a dict form of nodes in a tree: {id: node_instance}."""
        return self._nodes

    def parent(self, nid):
        """Get parent :class:`Node` object of given id."""
        if not self.contains(nid):
            raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)

        pid = self[nid].bpointer
        if pid is None or not self.contains(pid):
            return None

        return self[pid]

    def paths_to_leaves(self):

        res = []

        for leaf in self.leaves():
            res.append([nid for nid in self.rsearch(leaf.identifier)][::-1])

        return res

    def rsearch(self, nid, filter=None):
        """
        Traverse the tree branch along the branch from nid to its
        ancestors (until root).

        :param filter: the function of one variable to act on the :class:`Node` object.
        """
        if nid is None:
            return

        if not self.contains(nid):
            raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)

        filter = (self.__real_true) if (filter is None) else filter

        current = nid
        while current is not None:
            if filter(self[current]):
                yield current
            # subtree() hasn't update the bpointer
            current = self[current].bpointer if self.root != current else None

    def show(self, nid=None, level=ROOT, idhidden=True, filter=None,
             key=None, reverse=False, line_type='ascii-ex', data_property=None):

        self._reader = ""

        def write(line):
            self._reader += line.decode('utf-8') + "\n"

        try:
            self.__print_backend(nid, level, idhidden, filter,
                                 key, reverse, line_type, data_property, func=write)
        except NodeIDAbsentError:
            print('Tree is empty')

        print(self._reader)

    def getShow(self, nid=None, level=ROOT, idhidden=True, filter=None,
                key=None, reverse=False, line_type='ascii-ex', data_property=None):

        self._reader = ""

        def write(line):
            self._reader += line.decode('utf-8') + "\n"

        try:
            self.__print_backend(nid, level, idhidden, filter,
                                 key, reverse, line_type, data_property, func=write)
        except NodeIDAbsentError:
            print('Tree is empty')

        return self._reader

    def siblings(self, nid):

        siblings = []

        if nid != self.root:
            pid = self[nid].bpointer
            siblings = [self[i] for i in self[pid].fpointer if i != nid]

        return siblings

    def subtree(self, nid):

        st = Tree()
        if nid is None:
            return st

        if not self.contains(nid):
            raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)

        st.root = nid
        for node_n in self.expand_tree(nid):
            st._nodes.update({self[node_n].identifier: self[node_n]})
        return st

    def to_dict(self, nid=None, key=None, sort=True, reverse=False, with_data=False):
        """Transform the whole tree into a dict."""

        nid = self.root if (nid is None) else nid
        ntag = self[nid].tag
        tree_dict = {ntag: {"children": []}}
        if with_data:
            tree_dict[ntag]["data"] = self[nid].data

        if self[nid].expanded:
            queue = [self[i] for i in self[nid].fpointer]
            key = (lambda x: x) if (key is None) else key
            if sort:
                queue.sort(key=key, reverse=reverse)

            for elem in queue:
                tree_dict[ntag]["children"].append(
                    self.to_dict(elem.identifier, with_data=with_data, sort=sort, reverse=reverse))
            if len(tree_dict[ntag]["children"]) == 0:
                tree_dict = self[nid].tag if not with_data else \
                    {ntag: {"data": self[nid].data}}
            return tree_dict

    def to_json(self, with_data=False, sort=True, reverse=False):
        """To format the tree in JSON format."""
        return json.dumps(self.to_dict(with_data=with_data, sort=sort, reverse=reverse))
