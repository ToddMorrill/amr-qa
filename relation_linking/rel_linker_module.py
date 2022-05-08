# Interface through which relation linking generates candidates with scores
class Rel_linker:
    """
    Basic Interface for Relation Linkers
    """
    def __init__(self, config):
        pass

    def get_relation_candidates(self, question: str, params=None):
        """
        Takes in a question string and returns top-K candidate DBpedia relations
        :param text
        :param params: needs to contain the edge_component (e.g. 'die-01')
        :return: Counter a counter where the key is the candidate relation and count is the score
        """
        raise NotImplementedError('Please implement Relation Linker module')