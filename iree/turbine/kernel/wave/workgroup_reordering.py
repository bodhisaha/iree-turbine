from .._support.tracing import CapturedTrace
from ...support.logging import get_logger
#from .._support.indexing import IndexingContext
from .._support.indexing import *
from ..ops.wave_ops import *
from ..lang.global_symbols import *
#from .constraints import Constraint, get_constrained_shape
from .constraints import *
#from .utils.symbol_utils import subs_idxc
from .utils.symbol_utils import *
from .utils.graph_utils import move_node_after
from .utils.classes import KernelLaunchInfo
import math
import sympy

def reposition(node, new_wg0, new_wg1):
    wg0, wg1 = WORKGROUP_0, WORKGROUP_1
    for dim, symb_exp in node.index.items():
        symbols = symb_exp.start.free_symbols
        if wg0 in symbols:
            symb_exp.start = safe_subs(symb_exp.start, {wg0: new_wg0})
            continue
        elif wg1 in symbols:
            symb_exp.start = safe_subs(symb_exp.start, {wg1: new_wg1})
            continue



def reorder_workgroups(graph: CapturedTrace, workgroup_constraints):
    wg0, wg1 = [c.wg_dim for c in workgroup_constraints]
    num_wg_0, num_wg_1 = [c.count for c in workgroup_constraints]

    #flatten workgroup index
    flat_wg_index = wg1 * num_wg_0 + wg0
    GROUP_SIZE_1 = 2
    #num_wg_group is how many workgroups are in each group
    num_wg_group = GROUP_SIZE_1 * num_wg_0
    group_id = flat_wg_index // num_wg_group
    first_wg_id_1 = group_id * GROUP_SIZE_1

    new_wg0 = (flat_wg_index % num_wg_group) // GROUP_SIZE_1
    new_wg1 = first_wg_id_1 + (flat_wg_index % num_wg_group) % GROUP_SIZE_1

    #walk through the graph and call the reordering function on the operations
    graph_nodes = graph.walk()
    for node in graph_nodes:
        custom_node = get_custom(node)
        if (custom_node.index):
            op_set = {"iterate", "get_result"}
            if custom_node.name not in op_set:
                reposition(custom_node, new_wg0, new_wg1)
