# Interface through which relation linking generates candidates with scores
class Rel_linker:
    """
    Basic Interface for Relation Linkers
    """
    def __init__(self, config):
        pass

    def get_relation_candidates(self, params=None):
        """
        Takes in params and returns top-K candidate DBpedia relations
        :param params: needs to contain the edge_component (e.g. 'die-01') and input question string
        :return: Counter a counter where the key is the candidate relation and count is the score
        """
        raise NotImplementedError('Please implement Relation Linker module')