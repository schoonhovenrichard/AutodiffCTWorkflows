import itertools
from abc import ABC, abstractmethod
from collections import defaultdict

import dill as pickle
import torch


class BasePipeline(ABC):
    def __init__(self, operators):
        self.operators = operators

    @abstractmethod
    def __call__(self, data=None):
        pass

    @abstractmethod
    def get_dimensions(self, input_size):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    def to_device(self, device):
        for operator in self.operators:
            operator.to_device(device)

    def get_parameters(self):
        parameters = []
        for op in self.operators:
            if op.parameters is None:
                continue
            for par in op.parameters.values():
                parameters.append(par)
        return parameters

    def validate_dimensions(self, input_size):
        dimensions = self.get_dimensions(input_size)
        for i, dims in enumerate(dimensions[:-1]):
            op = self.operators[i]
            required_dims = op.required_input_dimensions()
            for j in range(len(required_dims)):
                if required_dims[j] is float('inf'):
                    break
                elif required_dims[j] is None:
                    continue
                elif required_dims[j] != dims[j]:
                    raise Exception(f"Mismatch dimensions at operator {j}")


class SequentialPipeline(BasePipeline):
    def __call__(self, data=None):
        for op in self.operators:
            if op.implements_batching:
                if data is None:
                    data = op()
                else:
                    data = op(data)
            else:
                if data is None:
                    data = [op()]
                else:
                    data = [op(x) for x in data]
                data = torch.stack(data, dim=0)
        return data

    def get_dimensions(self, input_size):
        current_dim = input_size
        dimensions = [input_size]
        for operator in self.operators:
            current_dim = operator.get_output_dimensions(current_dim)
            if operator.implements_batching:
                current_dim = current_dim[1:]
            dimensions.append(current_dim)
        return dimensions

    def save(self, path):
        operator_states = [op.get_state() for op in self.operators]
        operator_classes = [type(op) for op in self.operators]
        state = {
            "states": operator_states,
            "classes": operator_classes
        }

        with open(path, 'wb') as f:
            pickle.dump(state,f)

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        operator_states = state["states"]
        operator_classes = state["classes"]

        self.operators = []
        for cls, op_state in zip(operator_classes, operator_states):
            op = cls.__new__(cls)
            op.__dict__.update(op_state)
            self.operators.append(op)


class GraphPipeline(BasePipeline):
    def __init__(self, operators: list):
        super().__init__(operators)
        self.pipeline_graph = _GraphPipeline(operators)
        self.operators = self.pipeline_graph.operators
        self.intermediate_out = [None] * len(self.pipeline_graph)

    def _free_intermediate_outputs(self):
        for tensor in self.intermediate_out:
            del tensor
        self.intermediate_out = [None] * len(self.pipeline_graph)

    def __call__(self, data=None):
        self._free_intermediate_outputs()

        # A graph pipeline always has a single source and single sink.
        self.intermediate_out[0] = data

        # intermediate_outputs contains the (stacked) tensor output of all
        # operators that provide input for the i-th operator.
        for idx, op in enumerate(self.pipeline_graph):# Graph is iterable
            inp_data = self.intermediate_out[idx]

            # Get output tensor for this operator
            if op.implements_batching:
                if inp_data is None:
                    out_data = op()
                else:
                    out_data = op(inp_data)
            else:
                if inp_data is None:
                    out_data = [op()]
                else:
                    out_data = [op(x) for x in inp_data]

                if out_data[0] is not None:
                    out_data = torch.stack(out_data, dim=0)
                else:
                    out_data = None

            # Pass output data to relevant operators in the graph.
            for out_i in self.pipeline_graph.graph[idx]:
                if self.intermediate_out[out_i] is None:
                    self.intermediate_out[out_i] = out_data
                else:
                    # We assume the operator takes input as (batch_size, channels, ...)
                    # where channels is > 1 if the operator has more than 1 input
                    # operator.
                    if self.intermediate_out[out_i].ndim == out_data.ndim:
                        # both do not have channel dimension
                        self.intermediate_out[out_i] = torch.cat(
                            [self.intermediate_out[out_i].unsqueeze(1),
                            out_data.unsqueeze(1)], dim=1)
                    else:
                        self.intermediate_out[out_i] = torch.cat(
                            [self.intermediate_out[out_i], out_data.unsqueeze(1)], dim=1)
        self._free_intermediate_outputs()
        return out_data

    def get_dimensions(self, input_size):
        # for source vertex
        dimensions = [input_size]
        for k in range(self.pipeline_graph.num_vertices - 1):
            dimensions.append([])

        for idx, op in enumerate(self.pipeline_graph):# Graph is iterable
            input_dims = dimensions[idx]
            if type(input_dims) is list:
                #TODO: HANDLE MULTIPLE INPUT DIM TUPLES
                new_dims = []
                transpose_dims = list(map(list,
                                          itertools.zip_longest(*input_dims,
                                                                fillvalue=None)))
                for k, channel_dims in enumerate(transpose_dims):
                    channel_dims = [val for val in channel_dims if val != float('inf')]
                    if len(channel_dims) == 0:# They were all float(inf)
                        new_dims.append(float('inf'))
                        continue
                    channel_dims = [val for val in channel_dims if val is not None]
                    if len(channel_dims) == 0:# They were all float(inf)
                        new_dims.append(None)
                        continue

                    # Now only numbers are left, i.e., the exact tensor dimensions for
                    # inputs along this channel. These need to be equal unless it is
                    # specifically the channel dimension. Since there is no batching,
                    # this is the first channel always.
                    if k == 0:
                        new_dims.append(sum(channel_dims))
                    else:
                        # If not all equal, throw error
                        all_equal = channel_dims.count(channel_dims[0])==len(channel_dims)
                        if all_equal:
                            new_dims.append(channel_dims[0])
                        else:
                            raise Exception("Input dimension to operator", idx, op, "not"
                                            " resolvable along channel dim", k, "with"
                                            " sizes", channel_dims, "at k-th channel.")

            out_dims = op.get_output_dimensions(input_dims)
            if op.implements_batching:
                out_dims = out_dims[1:]

            for out_i in self.pipeline_graph.graph[idx]:
                dimensions[out_i].append(out_dims)
        return dimensions

    def save(self, path):
        operator_states = [op.get_state() for op in self.operators]
        operator_classes = [type(op) for op in self.operators]
        state = {
            "states": operator_states,
            "classes": operator_classes,
            "graph": self.pipeline_graph
        }

        with open(path, 'wb') as f:
            p = pickle.Pickler(f)
            p.dump(state)

    def load(self, path):
        with open(path, 'rb') as f:
            up = pickle.Unpickler(f)
            state = up.load()
        operator_states = state["states"]
        operator_classes = state["classes"]

        self.operators = []
        for cls, op_state in zip(operator_classes, operator_states):
            op = cls.__new__(cls)
            op.__dict__.update(op_state)
            self.operators.append(op)

        self.pipeline_graph = state["graph"]
        self.intermediate_out = [None] * len(self.pipeline_graph)


class _GraphPipeline:
    def __init__(self, adj_operator_list):
        r"""
        Args:
            adj_operator_list (list(tuple(op, list))): A list of tuples, where
                the first element of each tuple is an operator, and the second
                element is a list of operators that take input from the first
                operator.
        """
        self.adj_operator_list = adj_operator_list
        self._build_graph()

        #TODO: Check if connected graph

        if self.is_cyclic():
            raise Exception("Cyclic pipeline detected. Pipelines must be"
                            " directed acyclic graphs (DAGs).")

        if self.has_multiple_sinks():
            raise Exception("Detected multiple sinks in the pipeline graph,"
                            " the pipeline must end in a single sink.")

        if self.has_multiple_sources():
            raise Exception("Detected multiple sources in the pipeline graph,"
                            " the pipeline must begin with a single source.")

    def __iter__(self):
        ''' Returns the Iterator object '''
        return iter(self.operators)

    def __len__(self):
        return self.num_vertices

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def _build_graph(self):
        self.operators = []
        self.receiving_ops = {}
        if len(self.adj_operator_list) == 0:
            self.is_empty_graph = True
            self.num_vertices = 0
            self.graph = defaultdict(list)
            return

        self.is_empty_graph = False

        for adj_tuple in self.adj_operator_list:
            op = adj_tuple[0]
            if op not in self.operators:
                self.operators.append(op)
                self.receiving_ops[op] = adj_tuple[1]
            else:
                self.receiving_ops[op] += adj_tuple[1]
        self.num_vertices = len(self.operators)
        self.graph = defaultdict(list)

        # Add edges between operators in the graph
        for idx in range(self.num_vertices):
            #operator = self.adj_operator_list[idx][0]
            #receiving_ops = self.adj_operator_list[idx][1]
            operator = self.operators[idx]
            receiving_ops = self.receiving_ops[operator]

            # Lookup the indices of the receiving operators
            node_idx = idx
            output_idxs = [self.operators.index(op) for op in self.receiving_ops[operator]]

            # Add the edges to the graph
            for out_idx in output_idxs:
                self.add_edge(node_idx, out_idx)

    def has_multiple_sinks(self):
        nr_sinks = 0
        for vertex in self.graph.keys():
            if len(self.graph[vertex]) == 0:
                nr_sinks += 1

        if self.is_empty_graph:
            return False
        if nr_sinks == 0:
            raise Exception("Error in graph algorithm, acyclic graph must have"
                            " at least one sink.")
        if nr_sinks == 1:
            return False
        else:
            return True

    def has_multiple_sources(self):
        has_incoming_edges = [False] * self.num_vertices
        vertices = list(self.graph.keys())
        for v_i in vertices:
            outgoing_edges = self.graph[v_i]
            for target_node in outgoing_edges:
                has_incoming_edges[target_node] = True

        # The nodes without incoming edges are sources
        nr_sources = self.num_vertices - sum(has_incoming_edges)

        if self.is_empty_graph:
            return False
        if nr_sources == 0:
            raise Exception("Error in graph algorithm, acyclic graph must have"
                            " at least one source.")
        if nr_sources == 1:
            return False
        else:
            return True

    def is_cyclic(self):
        r""" We can detect cycles by running a full DFS, getting the search
         tree, and checking if there is a back edge in the tree, i.e., an edge
         from a node to itself or an ancestor.

        Return True if graph is cyclic, else False.
        """
        visited = [False] * (self.num_vertices + 1)
        recursion_stack = [False] * (self.num_vertices + 1)
        for vertex_idx in range(self.num_vertices):
            if not visited[vertex_idx]:
                if self._cyclic_helper(vertex_idx, visited, recursion_stack):
                    return True
        return False

    def _cyclic_helper(self, v_i, visited, rec_stack):
        # Currect node is marked as visited
        visited[v_i] = True
        rec_stack[v_i] = True

        # Check all neigbours, if any are visited, graph is cyclic.
        for neighbour in self.graph[v_i]:
            if not visited[neighbour]:
                if self._cyclic_helper(neighbour, visited, rec_stack):
                    return True
            elif rec_stack[neighbour]:
                return True

        # Pop the visited vertex from the recustion stack
        rec_stack[v_i] = False
        return False
