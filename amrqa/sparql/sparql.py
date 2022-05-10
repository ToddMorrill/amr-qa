"""This module transforms text questions into SPARQL queries.

Examples:
    $ python -m amrqa.sparql.sparql \
        --data-filepath ./amrqa/sparql/qald_9.json \
        --fast-align-dir /home/iron-man/Documents/fast_align/build \
        --propbank-filepath /home/iron-man/Documents/data/amr-qa/probbank-dbpedia.pkl \
        --index 339
    
    $ python -m amrqa.sparql.sparql \
        --data-filepath ./amrqa/sparql/qald_9.json \
        --fast-align-dir ~/Documents/fast_align/build \
        --propbank-filepath ~/Documents/data/amr-qa/probbank-dbpedia.pkl \
        --index 254
    
    $ python -m amrqa.sparql.sparql \
        --data-filepath ./amrqa/sparql/qald_9.json \
        --fast-align-dir ~/Documents/fast_align/build \
        --propbank-filepath ~/Documents/data/amr-qa/probbank-dbpedia.pkl \
        --save-dir ~/Documents/data/amr-qa/generate/v2
"""
import argparse
import os

from amrlib.alignments.faa_aligner import FAA_Aligner
import networkx as nx
import penman
from transformers import BertTokenizer, BertModel

from ..relation_linking import bert_rel_linker, ask_validation
from . import utils
from .amr import AMR


class SPARQLConverter(object):
    """Converts AMR graphs to SPARQL queries."""

    def __init__(self, amr, propbank_mapping, relation_linker) -> None:
        self.amr = amr
        self.propbank_mapping = propbank_mapping
        self.propbank_predicates = set(
            list(self.propbank_mapping['relation_scores'].keys()))
        self.prefixes = {
            'http://dbpedia.org/ontology/': 'dbo:',
            'http://dbpedia.org/resource/': 'res:',
            'http://www.w3.org/2000/01/rdf-schema#': 'rdfs:'
        }
        self.relation_linker = relation_linker

    def _is_imperative(self):
        """Checks if the AMR graph is imperative based on the presence of an
        'imperative' node."""
        imperative = False
        for src, role, tgt in self.amr.penman_g.triples:
            if role == ':mode' and tgt == 'imperative':
                imperative = True
        return imperative

    def _remove_source_node_edges(self, source, graph):
        """Removes the root node of a graph along with all of its outbound and
        inbound edges. NB: this does not delete child nodes or their edges."""
        filtered_instances = []
        for triple in graph.instances():
            src, role, tgt = triple
            if (src != source):
                filtered_instances.append(triple)

        filtered_edges = []
        for triple in graph.edges():
            src, role, tgt = triple
            if (src != source) and (tgt != source):
                filtered_edges.append(triple)

        filtered_attributes = []
        for triple in graph.attributes():
            src, role, tgt = triple
            if (src != source) and (tgt != source):
                filtered_attributes.append(triple)

        # create new graph
        new_g = penman.Graph(filtered_instances + filtered_edges +
                             filtered_attributes)
        return new_g

    def _restructure_graph(self, g):
        """If the graph is imperative, then make the ARG1 node of the root node
        an 'amr-unknown' node. Then delete the source node and its edges. See
        Algorithm 1, lines 3-6 - https://arxiv.org/pdf/2012.01707.pdf"""
        # retrieve the ARG1 node from r
        r = g.top

        # retrieve ARG1 node of the source
        # TODO: what to do if ARG1 is not present?
        arg1 = None
        arg1s = [
            tgt for src, role, tgt in g.edges()
            if (role == ':ARG1') and (src == r)
        ]
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
            # make the node an instance of amr-unknown
            new_instance = penman.graph.Instance(source=instances[idx].source,
                                                 role=':instance',
                                                 target='amr-unknown')
            instances[idx] = new_instance
            # update the graph
            g = penman.graph.Graph(triples=g.edges() + instances +
                                   g.attributes())
            # delete r and it's edges
            g = self._remove_source_node_edges(g.top, g)
        return g

    def _handle_mod_edge(self, g):
        """This function detects if the 'amr-unknown' node is a child of a
        ':mod' edge, and if so, treats the parent as the 'amr-unknown' node.
        See Algorithm 1, lines 7-9 - https://arxiv.org/pdf/2012.01707.pdf."""
        # treat parent on mod edge as the amr-unknown
        # NB: we are assuming only one amr-unknown will be present
        a = 'amr-unknown'
        a_node_id = None
        a_node_ids = [
            src for src, role, tgt in g.instances()
            if (role == ':instance') and (tgt == a)
        ]
        if a_node_ids:
            a_node_id = a_node_ids[0]

        # if there is a modifier edge
        for src, role, tgt in g.edges() + g.attributes():
            # if (src == a_node_id) and (role == ':mod'):
            #     a_node_id = tgt
            #     break
            if (tgt == a_node_id) and (role == ':mod'):
                a_node_id = src
                break
        return a_node_id

    def _create_undirected_graph(self, g):
        """Creates an undirected networkx graph so that shortest paths between
        all entity nodes and the amr-unknown node can be computed."""
        G = nx.Graph()
        edge_list = []
        for src, role, tgt in g.edges():
            edge_list.append((src, tgt, {'role': role}))

        # filter attributes (i.e. ignore entity related attributes)
        for src, role, tgt in g.attributes():
            if role.startswith(':op'):
                # handle quoted strings (e.g. '"Air"')
                if tgt.startswith('"'):
                    tgt = tgt[1:-1]
                edge_list.append((src, tgt, {'role': role}))
        G.add_edges_from(edge_list)
        return G

    def _shortest_paths(self, g, G, entity_node_id):
        """Finds the shortest paths between all entity nodes and 'amr-unknown'
        node and creates query graph edges and nodes based on the path
        information. See lines 11-24 in Algorithm 1 -
        https://arxiv.org/pdf/2012.01707.pdf"""
        amr_path = []
        # amr_unkown_node_id may be None
        if self.amr_unknown_node_id is not None:
            # TODO: handle case where query is imperative and an entity node
            # gets put in its own disjoint graph component (e.g. 23, 33)
            # TODO: address nx.NetworkXNoPath for 54, 77, 177, 290
            amr_path = nx.shortest_path(G, self.amr_unknown_node_id,
                                        entity_node_id)
        collapsed_path = [self.amr_unknown_node_id]
        source = self.amr_unknown_node_id  # n' in Algorithm 1
        rel_builder = ''
        # TODO: if amr_path is empty, check if the question is of the form "Is horse racing a sport?"
        for idx, target in enumerate(amr_path[1:]):
            # get instance type of node
            node_types = [
                tgt for src, role, tgt in g.instances() if src == target
            ]
            if node_types:
                node_type = node_types[0]
            else:
                # e.g. for op{n} nodes such as 'Air'
                node_type = 'literal'
            if node_type in self.propbank_predicates:
                rel = ''
                # get relation type
                for src, role, tgt in g.edges():
                    if ((src == source) and
                        (tgt == target)) or ((src == target) and
                                             (tgt == source)):
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
                next_node_on_path = amr_path[1:][idx + 1]
                # get relation type
                for src, role, tgt in g.edges():
                    if ((src == next_node_on_path) and
                        (tgt == target)) or ((src == target) and
                                             (tgt == next_node_on_path)):
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
                self.query_nodes.add(source)
                self.query_nodes.add(target)  # not done in the paper
                # if rel_builder = '', get relation type
                if not rel_builder:
                    # if mod relation type, get target type as relation
                    relations = [
                        tgt for src, role, tgt in g.edges() + g.attributes()
                        if (role == ':mod')
                        & ((src == source) | (tgt == source))
                    ]
                    if relations:
                        # get instance type
                        rel_builder = [
                            tgt for src, role, tgt in g.instances()
                            if (src == relations[0])
                        ][0]
                # TODO: can we switch the order here to form a valid query?
                self.query_edges.add((target, rel_builder, source))
                source = target
                rel_builder = ''

    def _get_type_edges(self, g):
        """Adds "type" edges to the query edge set, which constrain the SPARQL
        where clause (e.g. m type movie)."""
        # handle case when amr-unknown is on a mod edge (need parent type)
        # TODO: address the imperative case - we're dropping the entity type (e.g. index 8)
        for node in self.query_nodes:
            entity_type = [
                tgt for src, role, tgt in g.instances()
                if (src == node) & (role == ':instance')
            ]
            if entity_type:
                entity_type = entity_type[0]
            else:
                # e.g. for op{n} nodes such as 'Air'
                entity_type = 'literal'
            # potentially modify entity_type if amr-unknown and there is a modifier edge
            if entity_type == 'amr-unknown':
                for src, role, tgt in g.edges() + g.attributes():
                    if ((tgt == node) | (src == node)) and (role == ':mod'):
                        entity_type = [
                            tgt_ for src_, role_, tgt_ in g.instances()
                            if ((src_ == src) | (src_ == tgt))
                            & (role_ == ':instance')
                        ][0]
                        break
            type_edge = (node, 'type', entity_type)
            self.query_edges.add(type_edge)

    def _get_query_graph(self, g, G):
        """This method generates the query graph, namely the nodes and edges,
        that will be converted into the SPARQL where clause."""
        self.query_nodes = set()
        self.query_edges = set()
        for entity_node_id in self.amr.entity_nodes:
            # may be None if entity node is mispelled
            # e.g. New Yor City
            if entity_node_id is None:
                continue

            self._shortest_paths(g, G, entity_node_id)

        self._get_type_edges(g)

    def _ground_edges(self):
        """This method takes the entity variable placeholders found on the
        query graph edges (variables are originally from the AMR graph) and
        grounds them to known entities. This method does Bert-based relation
        linking."""
        grounded_edges = set()
        for src, edge, tgt in self.query_edges:
            # look for entity
            source_entity = src
            target_entity = tgt
            if source_entity in self.amr.alignments:
                source_entity = self.amr.alignments[src]['dbpedia_entity']
            if target_entity in self.amr.alignments:
                target_entity = self.amr.alignments[src]['dbpedia_entity']

            # relation = None
            # for edge_component in edge.split('|'):
            #     if edge_component in self.propbank_mapping['relation_scores']:
            #         relations = self.propbank_mapping['relation_scores'][edge_component]
            #         if len(relations) > 0:
            #             relation = relations[0]['rel']

            # Bert-based relation linking
            relation_linker_params = {'question': self.amr.sentence, 'top-K': 1, 'threshold': 0.6, 'do_cap': False}
            relation = None
            for edge_component in edge.split('|'):
                if edge_component in self.propbank_mapping['relation_scores']:
                    relation_linker_params['edge_component'] = edge_component
                    relation = self.relation_linker.get_relation_candidates(
                        params=relation_linker_params)[
                            0]  # top-1 relation

            # use ASK queries to determine the order of the relation, may need to use _replace prefixes
            # TODO: address unlinked relations
            if relation != None:
                ask_validator = ask_validation.ASK_Validator()
                if ask_validator.validate(
                        self.prefixes,
                    (target_entity, relation, source_entity)) == 'true':
                    grounded_edges.add(
                        (target_entity, relation, source_entity))
                else:
                    grounded_edges.add(
                        (source_entity, relation, target_entity))

        return grounded_edges

    def _replace_prefixes(self, grounded_edges):
        """This method replaces DBPedia prefixes with commonly used
        placeholders. For example: http://dbpedia.org/resource/ -> res"""
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

    def algorithm_1(self):
        """Implements Algorithm 1 from https://arxiv.org/pdf/2012.01707.pdf,
        which is responsible for generating a query graph, which forms an
        intermediate representation that will be converted to the SPARQL where
        clause. This method also implements relation linking."""
        # line 3 of algorithm 1
        g = self.amr.penman_g
        if self._is_imperative():
            g = self._restructure_graph(g)

        # line 8 of algorithm 1
        self.amr_unknown_node_id = self._handle_mod_edge(g)

        # create undirected graph for shortest paths
        G = self._create_undirected_graph(g)

        # create the query graph
        self._get_query_graph(g, G)

        # do relation linking
        grounded_edges = self._ground_edges()
        self.grounded_edges = self._replace_prefixes(grounded_edges)

    def _get_query_components(self):
        """This method generates all substrings required to generate the SPARQL
        query."""
        # determine the query type
        # default to SELECT DISTINCT
        query_type = 'SELECT DISTINCT'

        # if no amr-unknown then ASK query
        a = 'amr-unknown'
        # a_node_id = None
        a_node_ids = [
            src for src, role, tgt in self.amr.penman_g.instances()
            if (role == ':instance') and (tgt == a)
        ]
        if not a_node_ids:
            query_type = 'ASK'
        # TODO: check logic for handling amr-unknown connected to polarity edges
        elif a_node_ids:
            a_node_id = a_node_ids[0]
            # amr-unknown connected to polarity edges
            for src, role, tgt in self.amr.penman_g.edges(
            ) + self.amr.penman_g.attributes():
                if ((src == a_node_id) or
                    (tgt == a_node_id)) and (role == ':polarity'):
                    query_type = 'ASK'
                    break

        # TODO: implement logic for ORDER BY
        # TODO: implement logic for counting (no QALD-9 queries require counting)

        # create the prefixes
        query_substrings = []
        for prefix in self.prefixes:
            query_substrings.append(
                f'PREFIX {self.prefixes[prefix]} <{prefix}>')

        query_substrings.append(query_type)

        # get the main variable of the query
        variable = None
        if hasattr(self, 'amr_unknown_node_id'):
            variable = f'?{self.amr_unknown_node_id}'

        # may be no variables
        if variable is not None:
            query_substrings.append(variable)
        query_substrings.append('WHERE')

        # generate the where clause
        triples = ['{']
        for element in self.grounded_edges:
            src, relation, tgt = element
            triples.append(f'{src} {relation} {tgt}.')
        triples.append('}')
        query_substrings.extend(triples)
        return query_substrings

    def generate_sparql(self):
        """Generates the SPARQL query for the specified AMR graph."""
        query_substrings = self._get_query_components()
        # TODO: how to handle all the possible 2^n possible orderings of triples
        sparql = ' '.join(query_substrings)
        return sparql


def generate_all(data, propbank_mapping, relation_linker, aligner=None):
    """Generates SPARQL queries from all the AMR graphs."""
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
            sparql = SPARQLConverter(example_amr, propbank_mapping, relation_linker)
            sparql.algorithm_1()
            query = sparql.generate_sparql()
            # TODO: add after fixing errors
            queries[key] = {'sparql': query, 'error': ''}
            query_edges[key] = {
                'query_edges': sorted(list(sparql.query_edges)),
                'error': ''
            }
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
    # BERT-based relation linker - only load once
    relation_linker_config = {
        'bert_model_type': 'bert-base-uncased',
        'do_lower_case': True,
        'bert_tokenizer_class': BertTokenizer,
        'bert_model': BertModel,
        'relation_scores': propbank_mapping['relation_scores']
    }
    relation_linker = bert_rel_linker.BERTRelationLinker(relation_linker_config)

    if args.index is not None:
        example = args.index
        sentence = qald[f'train_{example}']['text']
        amr = qald[f'train_{example}']['extended_amr']
        sample_amr = AMR(sentence=sentence, amr=amr, aligner=aligner)
        entity_nodes = sample_amr.get_entity_nodes()

        sparql = SPARQLConverter(sample_amr, propbank_mapping, relation_linker)
        sparql.algorithm_1()
        query = sparql.generate_sparql()
        print(sample_amr.sentence)
        print(qald[f'train_{example}']['sparql'])
        print(query)
    else:
        queries, query_edges, error_keys = generate_all(qald,
                                                        propbank_mapping,
                                                        relation_linker=relation_linker,
                                                        aligner=aligner)
        os.makedirs(args.save_dir, exist_ok=True)
        query_edges_filepath = os.path.join(args.save_dir, 'query_edges.json')
        gen_queries_filepath = os.path.join(args.save_dir,
                                            'generated_queries.json')
        utils.write_json(query_edges_filepath, query_edges)
        utils.write_json(gen_queries_filepath, queries)


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
    parser.add_argument('--index',
                        help='Example index within the data file to parse.')
    parser.add_argument(
        '--save-dir',
        help='Directory where the generated queries will be saved.')
    args = parser.parse_args()
    main(args)