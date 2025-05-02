# src/knowledge_graph/neo4j_interface.py
from neo4j import GraphDatabase

class Neo4jInterface:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        self.driver.close()
        
    def create_entity(self, entity_type, properties):
        """Create a node in the knowledge graph"""
        with self.driver.session() as session:
            # Create Cypher query
            query = f"CREATE (e:{entity_type} $props) RETURN e"
            
            # Execute query
            result = session.run(query, props=properties)
            return result.single()[0]
        
    def create_relation(self, source_id, relation_type, target_id, properties=None):
        """Create a relation between two nodes"""
        if properties is None:
            properties = {}
            
        with self.driver.session() as session:
            # Create Cypher query
            query = """
            MATCH (source) WHERE ID(source) = $source_id
            MATCH (target) WHERE ID(target) = $target_id
            CREATE (source)-[r:`{relation_type}` $props]->(target)
            RETURN r
            """.format(relation_type=relation_type)
            
            # Execute query
            result = session.run(query, source_id=source_id, target_id=target_id, props=properties)
            return result.single()[0]

    def add_text_entities(self, entities, document_id):
        """Add text entities to the knowledge graph"""
        entity_ids = {}

        for entity in entities:
            # Create entity node
            props = {
                "name": entity["text"],
                "source": "text",
                "document_id": document_id
            }

            node = self.create_entity(entity["type"], props)
            entity_ids[entity["text"]] = node.id

        return entity_ids

    def add_text_relations(self, relations, entity_ids):
        """Add text relations to the knowledge graph"""
        created_relations = []

        for relation in relations:
            source_id = entity_ids.get(relation["source"])
            target_id = entity_ids.get(relation["target"])

            if source_id and target_id:
                props = {
                    "sentence": relation["sentence"]
                }

                rel = self.create_relation(source_id, relation["relation"], target_id, props)
                created_relations.append(rel)

        return created_relations

    def add_image_objects(self, objects, image_id):
        """Add image objects to the knowledge graph"""
        object_ids = {}

        for obj in objects:
            # Create object node
            props = {
                "name": obj["label"],
                "confidence": obj["confidence"],
                "position": obj["position"],
                "source": "image",
                "image_id": image_id
            }

            node = self.create_entity(obj["type"], props)
            object_ids[obj["label"]] = node.id

        return object_ids

    def link_text_and_image(self, text_entity_ids, image_object_ids):
        """Link text entities with corresponding image objects"""
        links = []

        # Find matching entities by name
        for text_entity, text_id in text_entity_ids.items():
            for image_object, image_id in image_object_ids.items():
                # Simple string matching (can be improved with semantic matching)
                if text_entity.lower() in image_object.lower() or image_object.lower() in text_entity.lower():
                    # Create a link
                    rel = self.create_relation(text_id, "APPEARS_IN", image_id, {})
                    links.append(rel)

        return links

    def query_knowledge_graph(self, query, params=None):
        """Execute a custom Cypher query on the knowledge graph"""
        if params is None:
            params = {}

        with self.driver.session() as session:
            result = session.run(query, params)
            return [record for record in result]