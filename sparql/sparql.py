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
import os

import amrlib
from amrlib.alignments.faa_aligner import FAA_Aligner
import networkx as nx
import penman
from penman.models import noop

import utils
from utils import IndexBFS

class AMR(object):
    """Data container for the sentence, multiple versions of the AMR graph, and
    entities linked to AMR nodes."""
    def __init__(self, sentence=None, amr=None, aligner=None) -> None:
        if sentence is None and amr is None:
            raise ValueError(
                'At least one of "sentence" or "amr" must be passed.')
        self.sentence = sentence
        self.amr = amr
        self.aligner = aligner

        # parse sentence if AMR isn't provided
        if self.amr is None:
            stog = amrlib.load_stog_model()
            self.amr = stog.parse_sents([self.sentence])[0]
        # generate sentence if only AMR is provided
        elif self.sentence is None:
            gtos = amrlib.load_gtos_model()
            self.sentence = gtos.generate(graphs=[self.amr])[0]
        
        # create penman and networkx graphs of the amr
        self.penman_g, self.di_g = self._create_graph()

    def _create_graph(self):
        """Decodes the serialized AMR graph into a proper graph structure."""
        # decode using the NoOpModel so arg-of relations don't get reversed
        # this is necessary to index nodes properly for entity-to-AMR node alignment
        g = penman.decode(self.amr, noop.NoOpModel())
        edge_list = []
        for src, role, tgt in g.triples:
            # e.g. '"Abraham"'
            if tgt.startswith('"'):
                tgt = tgt[1:-1]
            # modify entity nodes so they don't conflict with AMR nodes
            # e.g. Skype in both entity list and core AMR graph 
            # ('all0', 'Skype', {'role': ':surface_form'})
            # and ('n', 'Skype', {'role': ':op1'})
            if role == ':surface_form':
                # will need to account for this later if using entity nodes
                tgt = f'{tgt}<surface form>'

            edge_list.append((src, tgt, {'role': role}))

        # create networkx graph
        di_g = nx.DiGraph()
        di_g.add_edges_from(edge_list)
        return g, di_g
    
    def _align_amr_tokens(self):
        """Aligns AMR nodes with sentence tokens."""
        # remove entity nodes from the graph to improve alignment quality
        split_amr = self.amr.split(':entities')
        if len(split_amr) == 2:
            amr_filtered = split_amr[0].strip() + ')'
        else:
            amr_filtered = split_amr[0]
        
        # transform to list, which is the expected input format
        sentence_lower = [self.sentence.lower()]
        amr_filtered = [amr_filtered]

        # align sentences with AMR nodes
        amr_surface_aligns, alignment_strings = self.aligner.align_sents(
            sentence_lower, amr_filtered)
        return amr_surface_aligns[0], alignment_strings[0]
    
    def _get_surface_forms(self):
        """Gets the surface forms correspond to the DBPedia entities provided
        in the AMR graph."""
        surface_forms = []
        for u, v, e in self.di_g.edges(data=True):
            if e['role'] == ':surface_form':
                # strip out identifying info (e.g. <surface form>)
                v = v.split('<surface form>')[0]
                surface_forms.append(v)
        return surface_forms
    
    def _surface_form_token_indexes(self):
        """Retrieves token positions in the sentence of the DBPedia entity
        surface forms."""
        token_idxs = []
        for surface_form in self.surface_forms:
            # TODO: handle cases where surface form doesn't appear in text (e.g typos, etc.)
            # e.g. New Yor City
            if surface_form not in self.sentence:
                token_idxs.append([None])
                continue
            # NB: assuming surface_form only appears once in the text
            start_idx = self.sentence.index(surface_form)
            # remove everything after (and including) surface form
            pre_surface_form = self.sentence[:start_idx]
            tokens_before = len(pre_surface_form.split())
            tokens_in_surface_form = [
                i + tokens_before for i in range(len(surface_form.split()))
            ]
            token_idxs.append(tokens_in_surface_form)

        return token_idxs
    
    def _link_entities_to_amr_nodes(self):
        """Links DBPedia entities to AMR nodes via the surface forms. In
        particular, this method first aligns AMR nodes to sentence tokens. Then
        it links DBPedia entities to sentence tokens, which completes the
        mapping between DBPedia entities and AMR nodes."""
        # align AMR nodes to sentence tokens
        alignments = self._align_amr_tokens()
        self.amr_surface_aligned, self.token_alignment = alignments

        # get surface forms for all DBPedia entities
        # NB: assuming the AMR graph is annotated with surface form nodes
        # TODO: adapt this to new entity format, as needed
        self.surface_forms = self._get_surface_forms()

        # identify the token index position of these surface forms in the sentence
        self.token_idxs = self._surface_form_token_indexes()

        # use alignment strings to retrieve a node from the graph
        # self.token_alignment is a string of the form
        # [0-1.1 1-1.1.2 ...], where the number before the dash is the token
        # index in the sentence, and the number after the dash is an index into
        # AMR graph where each decimal corresponds to a level in the tree and
        # the numbers correspond to the order of the child nodes
        alignments = {}
        for mapping in self.token_alignment.split():
            tok_idx, graph_idx = mapping.split('-')
            alignments[int(tok_idx)] = graph_idx
        self.alignments = alignments

        # retrieve graph_idxs for entity surface form tokens
        graph_idxs = []
        for surface_form in self.token_idxs:
            surface_form_g_idxs = []
            for token_idx in surface_form:
                if token_idx in alignments:
                    surface_form_g_idxs.append(alignments[token_idx])
            graph_idxs.append(surface_form_g_idxs)
        self.graph_idxs = graph_idxs

        # index the graph nodes according to the aligner
        idx_bfs = IndexBFS(self.di_g, root=self.penman_g.top)
        self.indexed_graph = idx_bfs.bfs()
    
    def _get_entity_nodes(self):
        """Retrieves the AMR nodes that have been marked as entities."""
        entity_nodes = []
        for entity_idxs in self.graph_idxs:
            parents = []
            for idx in entity_idxs:
                # filter the graph for the particular node we're looking for
                node = [node_ for node_, data in self.indexed_graph.nodes(data=True) if data['index'] == idx]
                # AMR entity nodes might not be found due to typos
                # e.g. New Yor City
                if node:
                    node = node[0]
                    try:
                        # get the parent of the entity node
                        parents.append(next(self.indexed_graph.predecessors(node)))
                    # catch case when node is root and there are no predecessors
                    except StopIteration:
                        pass
                else:
                    node = None
            
            if not parents:
                entity_nodes.append(None)
                continue
            
            # should assert that all surface form tokens (e.g. "New", "York",
            # "City") all share the same parent and determine what to do if they
            # don't all share the same parent
            # the parents may not be the same due to typos in the entities
            # all_same = all(x == parents[0] for x in parents)
            # assert all_same

            # naively take the parent of the first token
            parent = parents[0]
        
            # if parent is an instance of a name
            if ('name' in self.indexed_graph[parent]) and (self.indexed_graph[parent]['name']['role'] == ':instance'):
                # then retrieve the parent of 'name' and treat that as an entity node
                grandparent = next(self.indexed_graph.predecessors(parent))
                entity_nodes.append(grandparent)
            else:
                entity_nodes.append(node)
        return entity_nodes

    def _get_dbpedia_entities(self):
        """Retrieves the DBPedia entity resource for the surface forms in the AMR
        graph."""
        dbpedia_entities = []
        for surface_form in self.surface_forms:
            # recall that surface_form nodes were made unique
            surface_form = f'{surface_form}<surface form>'
            parent = [x for x in self.di_g.predecessors(surface_form)][0]
            for u, v, e in self.di_g.out_edges(parent, data=True):
                if e['role'] == ':uri':
                    dbpedia_entities.append(v)
        return dbpedia_entities

    def get_entity_nodes(self):
        # align DBPedia entities with AMR nodes
        self._link_entities_to_amr_nodes()
        # retrieve AMR nodes that have been marked as entities
        self.entity_nodes = self._get_entity_nodes()
        # retrieve the DBPedia entities for completeness
        self.dbpedia_entities = self._get_dbpedia_entities()

        # zip everything into one dictionary
        complete_alignments = {}
        complete_alignments['entity_nodes'] = self.entity_nodes
        complete_alignments['surface_forms'] = self.surface_forms
        complete_alignments['token_indexes'] = self.token_idxs
        complete_alignments['graph_indexes'] = self.graph_idxs
        complete_alignments['dbpedia_entities'] = self.dbpedia_entities
        self.complete_alignments = complete_alignments
        return complete_alignments

class SPARQLConverter(object):
    def __init__(self, amr, propbank_mapping) -> None:
        self.amr = amr
        self.propbank_mapping = propbank_mapping
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
                # TODO: handle case where query is imperative and an entity node
                # gets put in its own disjoint graph component (e.g. 23, 33)
                # TODO: address nx.NetworkXNoPath for 54, 77, 177, 290
                amr_path = nx.shortest_path(G, a_node_id, entity_node_id)
            collapsed_path = [a_node_id]
            source = a_node_id # n' in algo
            rel_builder = ''
            # TODO: if amr_path is empty, check if the question is of the form "Is horse racing a sport?"
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
            # TODO: use ASK queries to determine the order of the relation
            # TODO: address unlinked relations
            if relation != None:
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
        # default to SELECT DISTINCT
        query_type = 'SELECT DISTINCT'
        
        # if no amr-unknown then ASK query
        a = 'amr-unknown'
        # a_node_id = None
        a_node_ids = [src for src, role, tgt in self.amr.penman_g.instances() if (role == ':instance') and (tgt == a)]
        if not a_node_ids:
            query_type = 'ASK'
        # TODO: check logic for handling amr-unknown connected to polarity edges
        elif a_node_ids:
            a_node_id = a_node_ids[0]
            # amr-unknown connected to polarity edges
            for src, role, tgt in self.amr.penman_g.edges() + self.amr.penman_g.attributes():
                if ((src == a_node_id) or (tgt == a_node_id)) and (role == ':polarity'):
                    query_type = 'ASK'
                    break

        # TODO: confirm logic for handling the target variable is already covered in algo_1
        # TODO: implement logic for ORDER BY

        # TODO: implement logic for counting (no QALD-9 queries require counting)

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

def generate_all(data, propbank_mapping, aligner=None):
    queries = {}
    query_edges = {}
    error_keys = []
    for i in range(len(data)):
        sentence = data[f'train_{i}']['text']
        amr = data[f'train_{i}']['extended_amr']
        example_amr = AMR(sentence=sentence, amr=amr, aligner=aligner)
        entity_nodes = example_amr.get_entity_nodes()
        key = f'train_{i}'
        try:
            sparql = SPARQLConverter(example_amr, propbank_mapping)
            sparql.algorithm_1()
            query = sparql.generate_sparql()
            # TODO: add after fixing errors
            queries[key] = {'sparql': query, 'error': ''}
            query_edges[key] = {'query_edges': sorted(list(sparql.query_edges)), 'error': ''}
        except Exception as e:
            queries[key] = {'error': str(e)}
            query_edges[key] = {'error': str(e)}
            error_keys.append(key)
    return queries, query_edges, error_keys


def main(args):
    qald = utils.load_json(args.data_filepath)
    # set fast-aligner directory
    os.environ['FABIN_DIR'] = args.fast_align_dir
    aligner = FAA_Aligner()
    propbank_mapping = utils.load_pickle(args.propbank_filepath)

    if args.index is not None:
        example = args.index
        sentence = qald[f'train_{example}']['text']
        amr = qald[f'train_{example}']['extended_amr']
        sample_amr = AMR(sentence=sentence, amr=amr, aligner=aligner)
        entity_nodes = sample_amr.get_entity_nodes()
        
        sparql = SPARQLConverter(sample_amr, propbank_mapping)
        sparql.algorithm_1()
        query = sparql.generate_sparql()
        print(sample_amr.sentence)
        print(qald[f'train_{example}']['sparql'])
        print(query)
        breakpoint()
    else:
        queries, query_edges, error_keys = generate_all(qald, propbank_mapping, aligner=aligner)
        utils.write_json('query_edges.json', query_edges)
        utils.write_json('generated_queries.json', queries)

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