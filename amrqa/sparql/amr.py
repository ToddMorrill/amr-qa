import amrlib
import penman
from penman.models import noop
import networkx as nx

from .utils import IndexBFS


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
        amr_entity_alignments = {}
        for mapping in self.token_alignment.split():
            tok_idx, graph_idx = mapping.split('-')
            amr_entity_alignments[int(tok_idx)] = graph_idx
        self.amr_entity_alignments = amr_entity_alignments

        # retrieve graph_idxs for entity surface form tokens
        graph_idxs = []
        for surface_form in self.token_idxs:
            surface_form_g_idxs = []
            for token_idx in surface_form:
                if token_idx in self.amr_entity_alignments:
                    surface_form_g_idxs.append(
                        self.amr_entity_alignments[token_idx])
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
                node = [
                    node_
                    for node_, data in self.indexed_graph.nodes(data=True)
                    if data['index'] == idx
                ]
                # AMR entity nodes might not be found due to typos
                # e.g. New Yor City
                if node:
                    node = node[0]
                    try:
                        # get the parent of the entity node
                        parents.append(
                            next(self.indexed_graph.predecessors(node)))
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
            if ('name' in self.indexed_graph[parent]) and (
                    self.indexed_graph[parent]['name']['role'] == ':instance'):
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
        alignments = {}
        for idx, entity_node in enumerate(self.entity_nodes):
            sub_dict = {}
            sub_dict['surface_form'] = self.surface_forms[idx]
            sub_dict['token_indexes'] = self.token_idxs[idx]
            sub_dict['graph_indexes'] = self.graph_idxs[idx]
            sub_dict['dbpedia_entity'] = self.dbpedia_entities[idx]
            alignments[entity_node] = sub_dict
        self.alignments = alignments
        return alignments