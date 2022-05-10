from SPARQLWrapper import SPARQLWrapper, XML
import xml.etree.ElementTree as ET


class ASKValidator(object):
    """Determines the appropriate ordering of triples (i.e.
    (entity1 relation entity2) or (entity2 relation entity1))."""

    def __init__(self) -> None:
        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    def validate(self, prefixes, edge_triple):
        """Returns the correct order of the edge per DBPedia relations:
        (source_entity, relation, target_entity)."""
        entity1, relation, entity2 = edge_triple

        # clean up the format of entities
        for prefix in prefixes:
            clean_entity1 = f'?{entity1}'
            for prefix in prefixes:
                if entity1.startswith(prefix):
                    clean_entity1 = "<{}>".format(entity1)
                    break

            clean_entity2 = f'?{entity2}'
            for prefix in prefixes:
                if entity2.startswith(prefix):
                    clean_entity2 = "<{}>".format(entity2)
                    break

        self.sparql.setQuery("""
            ASK WHERE {{
                {} {} {}.
            }}
        """.format(clean_entity1, relation, clean_entity2))

        self.sparql.setReturnFormat(XML)
        results = self.sparql.query().convert()
        xml_results = results.toxml()
        root = ET.fromstring(xml_results)
        for child in root.findall("*"):
            if 'boolean' in child.tag:
                is_correct_order = child.text

        return is_correct_order


if __name__ == "__main__":
    prefixes = {
        'http://dbpedia.org/ontology/': 'dbo:',
        'http://dbpedia.org/resource/': 'res:',
        'http://www.w3.org/2000/01/rdf-schema#': 'rdfs:'
    }

    # edge_triple = ('a', 'dbo:deathPlace', 'http://dbpedia.org/resource/Abraham_Lincoln')
    edge_triple = ('http://dbpedia.org/resource/Abraham_Lincoln',
                   'dbo:deathPlace', 'a')

    ask_validator = ASKValidator()
    if ask_validator.validate(prefixes, edge_triple) == 'false':
        print('not a valid order')
    else:
        print('order is valid')