import json
import pickle

def load_json(filepath):
    """Reads the JSON file specified into a dictionary."""
    with open(filepath, 'r') as f:
        return json.load(f)

def write_json(filepath, obj):
    """Writes the dictionary to the filepath."""
    with open(filepath, 'w') as f:
        return json.dump(obj, f)

def load_pickle(filepath):
    """Loads and returns the pickle object."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

class IndexBFS(object):
    """BFS style traversal from the root node of a graph in order to number
    nodes."""
    def __init__(self, g, root):
        self.g = g
        self.root = root
        # index the nodes
        self.g.nodes[root]['index'] = '1'
        self.visited = set()
        self.frontier = [root]

    def bfs(self):
        """Runs breadth-first search from the root node and indexes nodes."""
        while self.frontier:
            node = self.frontier.pop(0)
            if node in self.visited:
                continue
            else:
                self.visited.add(node)

            child_counter = 1
            for child in self.g.successors(node):
                # set index of children
                # no index required if edge type is instance
                if self.g[node][child]['role'] == ':instance':
                    self.g.nodes[child]['index'] = None
                    continue
                # if index already set, don't set it again
                # needed to avoid reindexing reentrant nodes
                if 'index' not in self.g.nodes[child]:
                    parent_index = self.g.nodes[node]['index']
                    child_index = f'{parent_index}.{child_counter}'
                    self.g.nodes[child]['index'] = child_index
                
                self.frontier.append(child)
                child_counter += 1
        return self.g