from collections import deque, defaultdict

class OrderedGraph:
    def __init__(self, max_nodes=None):
        self.max_nodes = max_nodes
        self.nodes = deque()
        self.edges = defaultdict(set)

    def add_node(self, node):
        if self.max_nodes and len(self.nodes) >= self.max_nodes:
            oldest = self.nodes.popleft()
            self.remove_node(oldest)
        self.nodes.append(node)
        self.edges.setdefault(node, set())

    def remove_node(self, node):
        if node in self.edges:
            for neighbor in self.edges[node]:
                self.edges[neighbor].discard(node)
            del self.edges[node]
        if node in self.nodes:
            self.nodes.remove(node)

    def pop_node(self):
        """Remove and return the oldest node"""
        if not self.nodes:
            return None
        oldest = self.nodes.popleft()
        if oldest in self.edges:
            for neighbor in self.edges[oldest]:
                self.edges[neighbor].discard(oldest)
            del self.edges[oldest]
        return oldest

    def add_edge(self, u, v):
        if u in self.edges and v in self.edges:
            self.edges[u].add(v)
            self.edges[v].add(u)

    def dfs(self, node, ret_visited = True):
        visited = set()
        stack = [node]
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                for neighbor in self.edges[curr]:
                    if neighbor not in visited:
                        stack.append(neighbor)
        if ret_visited:
            return visited

    def connected_components(self):
        cmpts = []
        nodes_c = self.nodes.copy()
        while nodes_c:
            curr = nodes_c.popleft()
            cmpt = self.dfs(curr)
            cmpts.append(cmpt)
        return cmpts

    def __repr__(self):
        return f"OrderedGraph(nodes={list(self.nodes)}, edges={dict(self.edges)})"


class ENG(OrderedGraph, ):
    def __init__(self, epsilon, distance_fn, max_nodes = None):
        super().__init__(max_nodes)
        self.epsilon = epsilon
        self.distance_fn = distance_fn

    def add_node(self, node):
        super().add_node(node)
        for other in list(self.nodes):
            if other != node and self.distance_fn(node, other) < self.epsilon:
                #print(f"{node}-{other} added")
                self.add_edge(node, other)
