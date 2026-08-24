"""In-memory NetworkX graph with entity, relationship, and provenance helpers."""

from collections.abc import Iterator

import networkx as nx

from backend.graph_schema import GraphEntity, GraphRelationship, Provenance


class KnowledgeGraph:
    """Explainable directed graph backed by a lightweight in-memory MultiDiGraph."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self._entities: dict[str, GraphEntity] = {}
        self._relationships: dict[str, GraphRelationship] = {}

    def add_entity(self, entity: GraphEntity) -> GraphEntity:
        """Add an entity or merge new provenance into an existing entity."""

        existing = self._entities.get(entity.entity_id)
        if existing is None:
            self._entities[entity.entity_id] = entity
            self.graph.add_node(entity.entity_id, name=entity.name, entity_type=entity.entity_type)
            return entity
        for provenance in entity.provenance:
            if provenance not in existing.provenance:
                existing.provenance.append(provenance)
        return existing

    def add_relationship(self, relationship: GraphRelationship) -> GraphRelationship:
        """Add a provenance-bearing directed relationship."""

        self.graph.add_edge(
            relationship.source_entity,
            relationship.target_entity,
            key=relationship.relationship_id,
            relationship_id=relationship.relationship_id,
            relationship_type=relationship.relationship_type,
            provenance=relationship.provenance,
        )
        self._relationships[relationship.relationship_id] = relationship
        return relationship

    def entity(self, entity_id_or_name: str) -> GraphEntity | None:
        """Look up an entity by stable ID or display name."""

        normalized = entity_id_or_name.casefold()
        for entity in self._entities.values():
            if entity.entity_id == normalized or entity.name.casefold() == normalized:
                return entity
        return None

    def entities(self) -> list[GraphEntity]:
        return sorted(self._entities.values(), key=lambda item: item.name.casefold())

    def relationships(self, entity_id_or_name: str | None = None) -> list[GraphRelationship]:
        relationships = list(self._relationships.values())
        if entity_id_or_name is None:
            return relationships
        entity = self.entity(entity_id_or_name)
        if entity is None:
            return []
        return [
            relationship
            for relationship in relationships
            if relationship.source_entity == entity.entity_id or relationship.target_entity == entity.entity_id
        ]

    def neighbors(self, entity_id_or_name: str) -> list[GraphEntity]:
        entity = self.entity(entity_id_or_name)
        if entity is None:
            return []
        neighbor_ids = set(self.graph.successors(entity.entity_id)) | set(self.graph.predecessors(entity.entity_id))
        return [self._entities[entity_id] for entity_id in neighbor_ids]

    def paths(self, source: str, target: str, cutoff: int = 5) -> list[list[GraphRelationship]]:
        """Return simple directed paths represented as relationship sequences."""

        source_entity = self.entity(source)
        target_entity = self.entity(target)
        if source_entity is None or target_entity is None:
            return []
        found: list[list[GraphRelationship]] = []
        for node_path in nx.all_simple_paths(self.graph, source_entity.entity_id, target_entity.entity_id, cutoff=cutoff):
            relationship_path = []
            for left, right in zip(node_path, node_path[1:]):
                edge_data = next(iter(self.graph.get_edge_data(left, right).values()))
                relationship_path.append(self._relationships[edge_data["relationship_id"]])
            found.append(relationship_path)
        return found

    def provenance(self, entity_id_or_name: str) -> list[Provenance]:
        entity = self.entity(entity_id_or_name)
        return entity.provenance if entity else []

    def entity_type_counts(self) -> dict[str, int]:
        return _counts(entity.entity_type for entity in self._entities.values())

    def relationship_type_counts(self) -> dict[str, int]:
        return _counts(relationship.relationship_type for relationship in self._relationships.values())

    def __len__(self) -> int:
        return len(self._entities)

    @property
    def relationship_count(self) -> int:
        return len(self._relationships)


def _counts(values: Iterator[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
