"""This module transforms text questions into SPARQL queries.

Examples:
    $ python sparql.py \
        --data-filepath qald_9.json \
        --fast-align-dir /home/iron-man/Documents/fast_align/build \
        --propbank-filepath /home/iron-man/Documents/data/amr-qa/probbank-dbpedia.pkl \
        --index 339
    
    $ python sparql.py \
        --data-filepath qald_9.json \
        --fast-align-dir /home/iron-man/Documents/fast_align/build \
        --propbank-filepath /home/iron-man/Documents/data/amr-qa/probbank-dbpedia.pkl \
        --index 254
"""
import argparse
import json
import os
import pickle

import amrlib
from amrlib.alignments.faa_aligner import FAA_Aligner
import networkx as nx
import penman
from penman.models import noop


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# BFS style traversal from root node to number nodes
class IndexBFS(object):
    def __init__(self, g, root):
        self.g = g
        self.root = root
        # index the nodes
        self.g.nodes[root]['index'] = '1'
        self.visited = set()
        self.frontier = [root]

    def bfs(self):
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


class AMR(object):
    def __init__(self, sentence, amr) -> None:
        if sentence is None and amr is None:
            raise ValueError(
                'At least one of "sentence" or "amr" must be passed.')
        self.sentence = sentence
        self.amr = amr

        # parse sentence if AMR isn't provided
        if self.amr is None:
            stog = amrlib.load_stog_model()
            # TODO: determine if the return of parse_sents is subscriptable
            self.amr = stog.parse_sents([self.sentence])[0]
            # for graph in graphs:
            #     print(graph)
        # generate sentence if only AMR is provided
        elif self.sentence is None:
            gtos = amrlib.load_gtos_model()
            self.sentence = gtos.generate(graphs=[self.amr])[0]

    def create_graph(self):
        g = penman.decode(self.amr, noop.NoOpModel())
        self.penman_g = g
        edge_list = []
        # TODO: how to handle :ARG0-of relations
        # e.g. index 5 from QALD-9
        # https://penman.readthedocs.io/en/latest/api/penman.layout.html
        for src, role, tgt in g.triples:
            # e.g. '"Abraham"'
            if tgt.startswith('"'):
                tgt = eval(tgt)
            # modify entity nodes so they don't conflict with AMR nodes
            # e.g. Skype in both ('all0', 'Skype', {'role': ':surface_form'})
            # and ('n', 'Skype', {'role': ':op1'})
            if role == ':surface_form':
                # will need to account for this later
                tgt = f'{tgt}<surface form>'

            edge_list.append((src, tgt, {'role': role}))

        di_g = nx.DiGraph()
        di_g.add_edges_from(edge_list)
        return di_g

    def surface_form_indexes(self):
        word_idxs = []
        for surface_form in self.surface_forms:
            # assuming surface_form only appears once in the text
            # TODO: handle cases where surface form doesn't appear in text (e.g typos, etc.)
            # e.g. New Yor City
            if surface_form not in self.sentence:
                word_idxs.append([None])
                continue
            start_idx = self.sentence.index(surface_form)
            end_idx = start_idx + len(surface_form)
            # remove everything after (and including) surface form
            pre_surface_form = self.sentence[:start_idx]
            tokens_before = len(pre_surface_form.split())
            tokens_in_surface_form = [
                i + tokens_before for i in range(len(surface_form.split()))
            ]
            word_idxs.append(tokens_in_surface_form)

        self.word_idxs = word_idxs

    def link_surface_forms(self):
        inference = FAA_Aligner()
        # remove entity nodes from the graph to improve alignment quality
        split_amr = self.amr.split(':entities')
        if len(split_amr) == 2:
            amr_filtered = split_amr[0].strip() + ')'
        else:
            amr_filtered = split_amr[0]
        # transform to list, which is the expected input format
        sentence_lower = [self.sentence.lower()]
        amr_filtered = [amr_filtered]
        amr_surface_aligns, alignment_strings = inference.align_sents(
            sentence_lower, amr_filtered)
        self.amr_surface_aligned = amr_surface_aligns[0]
        self.token_alignment = alignment_strings[0]

        # use linked entities in AMR graph to retrieve surface forms
        # create networkx graph of the amr
        di_g = self.create_graph()
        # assuming the amr graph AMR graph is annotated with surface form nodes
        # TODO: adapt this to new entity format
        self.surface_forms = []
        for u, v, e in di_g.edges(data=True):
            if e['role'] == ':surface_form':
                # strip out unique info (e.g. <surface form>)
                v = v.split('<surface form>')[0]
                self.surface_forms.append(v)
        
        self.dbpedia_entities = []
        for surface_form in self.surface_forms:
            # recall that surface_form nodes were made unique
            surface_form = f'{surface_form}<surface form>'
            parent = [x for x in di_g.predecessors(surface_form)][0]
            for u, v, e in di_g.out_edges(parent, data=True):
                if e['role'] == ':uri':
                    self.dbpedia_entities.append(v)

        # identify the token index position of these surface forms in the sentence
        self.surface_form_indexes()

        # use alignment strings to narrow in on a node in the graph
        # use alignment strings to retrieve position in graph
        alignments = {}
        for mapping in self.token_alignment.split():
            tok_idx, graph_idx = mapping.split('-')
            alignments[int(tok_idx)] = graph_idx
        self.alignments = alignments

        # retrieve graph_idxs for surface form tokens
        graph_idxs = []
        for surface_form in self.word_idxs:
            surface_form_g_idxs = []
            for word_idx in surface_form:
                if word_idx in alignments:
                    surface_form_g_idxs.append(alignments[word_idx])
            graph_idxs.append(surface_form_g_idxs)

        self.graph_idxs = graph_idxs

        # index the graph nodes according to the aligner
        idx_bfs = IndexBFS(di_g, root=self.penman_g.top)
        self.indexed_graph = idx_bfs.bfs()
    
    def _get_entity_nodes(self):
        # check if parent is name type, and if so, retrieve the parent above name to treat as the entity node
        entity_nodes = []
        for entity_idxs in self.graph_idxs:
            parents = []
            for idx in entity_idxs:
                node = [x for x, y in self.indexed_graph.nodes(data=True) if y['index'] == idx][0]
                parents.append(next(self.indexed_graph.predecessors(node)))
            
            # AMR entity nodes might not be found due to typos
            # e.g. New Yor City
            if not parents:
                entity_nodes.append(None)
            else:
                # assert they all share the same parent
                all_same = all(x == parents[0] for x in parents)
                # TODO: what are the implications of this
                # e.g. QALD index 7 violates this 
                # assert all_same
                parent = parents[0]
            
                # if parent is an instance of a name
                if ('name' in self.indexed_graph[parent]) and (self.indexed_graph[parent]['name']['role'] == ':instance'):
                    # then retrieve the parent of 'name' and treat that as an entity node
                    grandparent = next(self.indexed_graph.predecessors(parent))
                    entity_nodes.append(grandparent)
                else:
                    entity_nodes.append(node)
            
            # TODO: if edge type is mod, retrieve parent to treat as the entity node
            # TODO: confirm if this is already handled by Algo 1
        
        self.entity_nodes = entity_nodes

    def get_entity_nodes(self):
        self.link_surface_forms()
        self._get_entity_nodes()
        # zip everything into one complete list
        self.complete_alignments = list(
            zip(self.entity_nodes, self.surface_forms, self.word_idxs, self.graph_idxs, self.dbpedia_entities))
        return self.entity_nodes

class SPARQLConverter(object):
    def __init__(self, amr, propbank_filepath) -> None:
        self.amr = amr
        self.propbank_mapping = load_pickle(propbank_filepath)
        self.propbank_predicates = set(list(self.propbank_mapping['relation_scores'].keys()))
        self.prefixes = {'http://dbpedia.org/ontology/': 'dbo:',
                    'http://dbpedia.org/resource/': 'res:',
                    'http://www.w3.org/2000/01/rdf-schema#': 'rdfs:'}
    
    def is_imperative(self):
        imperative = False
        for source, edge_label, dest in self.amr.penman_g.triples:
            if edge_label == ':mode' and dest == 'imperative':
                imperative = True
        return imperative
    
    def remove_source_node_edges(self, source, graph):
        """TODO: address children"""
        filtered_instances = []
        for instance in graph.instances():
            src, role, tgt = instance
            if src != source:
                filtered_instances.append(instance)
        
        filtered_edges = []
        for edge in graph.edges():
            src, role, tgt = edge
            if (src != source) and (tgt != source):
                filtered_edges.append(edge)
        
        filtered_attributes = []
        for attr in graph.attributes():
            src, role, tgt = attr
            if (src != source) and (tgt != source):
                filtered_attributes.append(attr)
                
        # create new graph
        new_g = penman.Graph(filtered_instances + filtered_edges + filtered_attributes)
        return new_g
    
    def restructure_graph(self, g):
        # retrieve the ARG1 node from r
        r = g.top
        # TODO: what to do if ARG1 is not present?
        arg1 = None
        arg1s = [tgt for src, role, tgt in g.edges() if (role == ':ARG1') and (src == r)]
        if arg1s:
            arg1 = arg1s[0]
            
        # get idx for row where source == arg1 and set it as amr-unknown
        found = False
        for idx, instance in enumerate(g.instances()):
            src, role, tgt = instance
            if src == arg1:
                found = True
                break
        if found:
            instances = g.instances()
            new_instance = penman.graph.Instance(source=src, role=role, target='amr-unknown')
            instances[idx] = new_instance
                # update the graph
            g = penman.graph.Graph(triples=g.edges()+instances+g.attributes())
                # delete r and it's edges
            g = self.remove_source_node_edges(g.top, g)
        return g
    
    def handle_mod_edge(self, g):
        # treat parent on mod edge as the amr-unknown
        # TODO: are we assuming only one amr-unknown will be present?
        a = 'amr-unknown'
        a_node_id = None
        a_node_ids = [src for src, role, tgt in g.instances() if (role == ':instance') and (tgt == a)]
        if a_node_ids:
            a_node_id = a_node_ids[0]
        # if there is a modifier edge
        for src, role, tgt in g.edges() + g.attributes():
            if (src == a_node_id) and (role == ':mod'):
                a_node_id = tgt
                break
            elif (tgt == a_node_id) and (role == ':mod'):
                a_node_id = src
                break
        return a_node_id
    
    def create_undirected_graph(self, g):
        G = nx.Graph()
        edge_list = []
        for src, role, tgt in g.edges():
            edge_list.append((src, tgt, {'role':role}))
        
        # filter attributes
        for src, role, tgt in g.attributes():
            if role.startswith(':op'):
                # handle quoted strings (e.g. '"Air"')
                if tgt.startswith('"'):
                    tgt = tgt[1:-1]
                edge_list.append((src, tgt, {'role':role}))
        G.add_edges_from(edge_list)
        return G
    
    def shortest_paths(self, g, G, a_node_id):
        query_nodes = set()
        query_edges = set()
        for entity_node_id in self.amr.entity_nodes:
            # may be None if entity node is mispelled
            # e.g. New Yor City 
            if entity_node_id is None:
                continue
            
            amr_path = []
            # a_node_id may be None
            if a_node_id is not None:
                amr_path = nx.shortest_path(G, a_node_id, entity_node_id)
            collapsed_path = [a_node_id]
            source = a_node_id # n' in algo
            rel_builder = ''
            for idx, target in enumerate(amr_path[1:]):
                # get instance type of node
                node_type = [tgt for src, role, tgt in g.instances() if src == target]
                if node_type:
                    node_type = node_type[0]
                else:
                    # e.g. for op{n} nodes such as 'Air'
                    node_type = 'literal'
                if node_type in self.propbank_predicates:
                    rel = ''
                    # get relation type
                    for src, role, tgt in g.edges():
                        if ((src == source) and (tgt == target)) or ((src == target) and (tgt == source)):
                            rel = role
                            break
                    # ignore core roles such as ARG{0,...,n}
                    if rel.startswith(':ARG'):
                        rel = ''
                    else:
                        # strip leading colon
                        rel = rel[1:] + '|'
                    
                    # retrieve the second part of the relation exiting from target
                    next_rel = ''
                    # TODO: will we hit index out-of-bounds? 
                    next_node_on_path = amr_path[1:][idx+1]
                    # get relation type
                    for src, role, tgt in g.edges():
                        if ((src == next_node_on_path) and (tgt == target)) or ((src == target) and (tgt == next_node_on_path)):
                            next_rel = role
                            break
                    # ignore core roles such as ARG{0,...,n}
                    if next_rel.startswith(':ARG'):
                        next_rel = ''
                    else:
                        next_rel = '|' + next_rel[1:]
                    
                    # equivalent to rel_builder + getRel
                    rel_builder = rel_builder + rel + node_type + next_rel
                # TODO: better understand def of A_c in algo 1
                else:
                    collapsed_path.append(target)
                    query_nodes.add(source)
                    query_nodes.add(target) # not done in the paper
                    # if rel_builder = '', get relation type
                    if not rel_builder:
                        # if mod relation type, get target type as relation
                        relations = [tgt for src, role, tgt in g.edges() + g.attributes() if (role == ':mod') & ((src == source) | (tgt == source))]
                        if relations:
                            # get instance type
                            rel_builder = [tgt for src, role, tgt in g.instances() if (src == relations[0])][0]
                    # TODO: can we switch the order here to form a valid query?
                    query_edges.add((target, rel_builder, source))
                    source = target
                    rel_builder = ''
        
        self.query_nodes = query_nodes
        self.query_edges = query_edges

        # TODO: address the imperative case
        # we're dropping the entity type (e.g. index 8)
        # TODO: breakout into new method
        # add "type" edges
        # handle case when amr-unknown is on a mod edge (need parent type)
        for node in self.query_nodes:
            entity_type = [tgt for src, role, tgt in g.instances() if (src == node) & (role == ':instance')]
            if entity_type:
                entity_type = entity_type[0]
            else:
                # e.g. for op{n} nodes such as 'Air'
                entity_type = 'literal'
            # potentially modify entity_type if amr-unknown and there is a modifier edge
            if entity_type == 'amr-unknown':
                for src, role, tgt in g.edges() + g.attributes():
                    if ((tgt == node) | (src == node)) and (role == ':mod'):
                        entity_type = [tgt_ for src_, role_, tgt_ in g.instances() if ((src_ == src) | (src_ == tgt)) & (role_ == ':instance')][0]
                        break
            type_edge = (node, 'type', entity_type)
            self.query_edges.add(type_edge)
        breakpoint()

    def ground_edges(self):
        grounded_edges = set()
        for src, edge, tgt in self.query_edges:
            # look for entity
            source_entity = src
            target_entity = tgt
            # TODO: implement a dictionary for this
            for item in self.amr.complete_alignments:
                node, surface_form, token_idxs, node_idxs, entity = item
                if src == node:
                    source_entity = entity
                elif tgt == node:
                    target_entity = entity
            
            # if edge contains a dbpedia predicate, greedily choose the top one
            # TODO: any reasonable defaults for relation?
            relation = None
            for edge_component in edge.split('|'):
                if edge_component in self.propbank_mapping['relation_scores']:
                    relations = self.propbank_mapping['relation_scores'][edge_component]
                    if len(relations) > 0:
                        relation = relations[0]['rel']
            grounded_edges.add((source_entity, relation, target_entity))
        return grounded_edges
    
    def replace_prefixes(self, grounded_edges):
        clean_grounded_edges = set()
        # replace prefixes
        for element in grounded_edges:
            src, relation, tgt = element
            clean_src = f'?{src}'
            for prefix in self.prefixes:
                if src.startswith(prefix):
                    clean_src = src.replace(prefix, self.prefixes[prefix])
                    break
            
            clean_tgt = f'?{tgt}'
            for prefix in self.prefixes:
                if tgt.startswith(prefix):
                    clean_tgt = tgt.replace(prefix, self.prefixes[prefix])
                    break
            
            clean_grounded_edges.add((clean_src, relation, clean_tgt))
        return clean_grounded_edges

    def generate_query(self):
        # determine the query type
        query_type = 'SELECT DISTINCT'
        # TODO: implement logic for ASK queries
        # TODO: implement logic for handling the target variable
        # TODO: implement logic for sorting
        # TODO: implement logic for counting

        # generate the query
        query_substrings = []
        for prefix in self.prefixes:
            query_substrings.append(f'PREFIX {self.prefixes[prefix]} <{prefix}>')

        query_substrings.append(query_type)

        # get the variable
        # TODO: what if there are 0 or more than 1 variables?
        variable = None
        for element in self.clean_grounded_edges:
            src, relation, tgt = element
            if src.startswith('?'):
                variable = src
            elif tgt.startswith('?'):
                variable = tgt
        
        # may be no variables
        if variable is not None:
            query_substrings.append(variable)
        query_substrings.append('WHERE')

        # prepare triples
        triples = ['{']
        for element in self.clean_grounded_edges:
            src, relation, tgt = element
            triples.append(f'{src} {relation} {tgt}.')
        triples.append('}')
        query_substrings.extend(triples)
        return query_substrings

    def generate_sparql(self):
        query_substrings = self.generate_query()
        # TODO: how to handle all the possible 2^n possible orderings of triples
        sparql = ' '.join(query_substrings)
        return sparql

    def algorithm_1(self):
        # AMR to query_graph
        # Algorithm 1: https://arxiv.org/pdf/2012.01707.pdf

        # if text is imperative
        # check if :mode imperative in the edges
        g = self.amr.penman_g
        if self.is_imperative():
            g = self.restructure_graph(g)

        a_node_id = self.handle_mod_edge(g)

        # create undirected graph for shortest paths
        G = self.create_undirected_graph(g)

        # create the query graph
        self.shortest_paths(g, G, a_node_id)

        # do relation linking
        # TODO: add instance types
        grounded_edges = self.ground_edges()
        self.clean_grounded_edges = self.replace_prefixes(grounded_edges)


def main(args):
    qald = load_json(args.data_filepath)
    # set fast-aligner directory
    os.environ['FABIN_DIR'] = args.fast_align_dir
    example = args.index
    sentence = qald[f'train_{example}']['text']
    amr = qald[f'train_{example}']['extended_amr']
    sample_amr = AMR(sentence=sentence, amr=amr)
    entity_nodes = sample_amr.get_entity_nodes()

    sparql = SPARQLConverter(sample_amr, args.propbank_filepath)
    sparql.algorithm_1()
    query = sparql.generate_sparql()
    print(sample_amr.sentence)
    print(qald[f'train_{example}']['sparql'])
    print(query)
    breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-filepath',
                        help='Filepath where the data is stored.')
    parser.add_argument(
        '--fast-align-dir',
        help='Directory where the fast-alignment executables were built.')
    parser.add_argument(
        '--propbank-filepath',
        help='Filepath where the PropBank to DBPedia mapping is stored.')
    parser.add_argument(
        '--index',
        help='Example index within the data file to parse.')
    args = parser.parse_args()
    main(args)