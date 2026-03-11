import re

from qdrant_client import QdrantClient, models

from engine.src.config import EngineConfig, CollectionVectorConfig


_DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "euclid": models.Distance.EUCLID,
    "dot": models.Distance.DOT,
}

_VERSION_SUFFIX_RE = re.compile(r"^(.+)_v(\d+)$")


class CollectionManager:
    """
    Manages Qdrant collection lifecycle with alias-based snapshot isolation.

    Collections use versioned names ({project}_v1, {project}_v2, ...) with an
    alias ({project}) pointing to the active version. Searches and reads go
    through the alias. Full re-indexes build a shadow collection and atomically
    swap the alias on completion.

    _client: QdrantClient — Active Qdrant client.
    _config: EngineConfig — Engine configuration with vector specs.
    _collection_name: str — Logical project name (used as alias).
    """

    def __init__(self, client: QdrantClient, config: EngineConfig, collection_name: str):
        """
        client: QdrantClient — Connected Qdrant client.
        config: EngineConfig — Engine config containing vector settings.
        collection_name: str — Logical project name (used as alias target).
        """
        self._client = client
        self._config = config
        self._collection_name = collection_name

    @property
    def client(self) -> QdrantClient:
        """
        Returns: QdrantClient — Active Qdrant client.
        """
        return self._client

    @property
    def collection_name(self) -> str:
        """
        Returns: str — Target collection name.
        """
        return self._collection_name

    def _build_vectors_config(self) -> dict[str, models.VectorParams]:
        """
        Build named vector config dict from EngineConfig.vectors.

        Returns: dict[str, VectorParams] — Map of vector name to params.
        """
        vectors = {}
        for vec in self._config.vectors:
            distance = _DISTANCE_MAP.get(vec.distance)
            if distance is None:
                raise ValueError(
                    f"Unknown distance metric: {vec.distance!r}. "
                    f"Supported: {list(_DISTANCE_MAP.keys())}"
                )
            vectors[vec.name] = models.VectorParams(
                size=vec.dimension,
                distance=distance,
            )
        return vectors

    def _collection_exists(self, name: str) -> bool:
        """
        Check if a specific collection name exists in Qdrant.

        name: str — Collection name to check.
        Returns: bool — True if collection exists.
        """
        try:
            self._client.get_collection(name)
            return True
        except Exception:
            return False

    def resolve_alias(self) -> str | None:
        """
        Resolve the alias to its underlying real collection name.

        Returns: str | None — Real collection name, or None if alias doesn't exist.
        """
        try:
            aliases = self._client.get_collection_aliases(self._collection_name)
            for alias in aliases.aliases:
                if alias.alias_name == self._collection_name:
                    return alias.collection_name
        except Exception:
            pass

        all_aliases = self._client.get_aliases()
        for alias in all_aliases.aliases:
            if alias.alias_name == self._collection_name:
                return alias.collection_name
        return None

    def active_collection(self) -> str:
        """
        Get the real collection name that the alias currently points to.
        Falls back to the logical name if no alias exists (legacy mode).

        Returns: str — Real collection name backing the alias.
        """
        resolved = self.resolve_alias()
        if resolved is not None:
            return resolved
        if self._collection_exists(self._collection_name):
            return self._collection_name
        return self._collection_name

    def _next_version(self) -> int:
        """
        Determine the next version number for a shadow collection.
        Scans existing {project}_v{N} collections and returns max(N) + 1.

        Returns: int — Next version number (starts at 1).
        """
        prefix = self._collection_name + "_v"
        max_version = 0
        collections = self._client.get_collections().collections
        for col in collections:
            if col.name.startswith(prefix):
                suffix = col.name[len(prefix):]
                if suffix.isdigit():
                    max_version = max(max_version, int(suffix))
        return max_version + 1

    def _latest_versioned_collection(self) -> str | None:
        """
        Find the highest existing versioned collection for this project.

        Returns: str | None — Latest versioned collection name, if any.
        """
        prefix = self._collection_name + "_v"
        latest_name = None
        latest_version = -1

        collections = self._client.get_collections().collections
        for col in collections:
            if not col.name.startswith(prefix):
                continue
            suffix = col.name[len(prefix):]
            if not suffix.isdigit():
                continue
            version = int(suffix)
            if version > latest_version:
                latest_version = version
                latest_name = col.name

        return latest_name

    def exists(self) -> bool:
        """
        Check if the project has an active collection (via alias or direct name).

        Returns: bool — True if a usable collection exists.
        """
        if self.resolve_alias() is not None:
            return True
        if self._latest_versioned_collection() is not None:
            return True
        return self._collection_exists(self._collection_name)

    def _create_versioned(self, version_name: str) -> None:
        """
        Create a versioned collection with vector config and payload indexes.

        version_name: str — Full versioned collection name (e.g., "project_v1").
        """
        self._client.create_collection(
            collection_name=version_name,
            vectors_config=self._build_vectors_config(),
        )
        self._create_payload_indexes_for(version_name)

    def create(self) -> bool:
        """
        Create the collection with alias-based versioning.
        Creates {project}_v1 and points alias {project} to it.
        No-op if alias or collection already exists.

        Returns: bool — True if created, False if already existed.
        """
        if self.exists():
            return False

        version_name = f"{self._collection_name}_v1"
        self._create_versioned(version_name)

        self._client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=version_name,
                        alias_name=self._collection_name,
                    )
                )
            ]
        )
        return True

    def ensure(self) -> bool:
        """
        Ensure the project has an active collection with alias.
        Migrates legacy (non-aliased) collections to the alias scheme.

        Returns: bool — True if collection was created or migrated, False if already set up.
        """
        if self.resolve_alias() is not None:
            return False

        latest_versioned = self._latest_versioned_collection()
        if latest_versioned is not None:
            self._client.update_collection_aliases(
                change_aliases_operations=[
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(
                            collection_name=latest_versioned,
                            alias_name=self._collection_name,
                        )
                    )
                ]
            )
            return True

        if self._collection_exists(self._collection_name):
            self._migrate_legacy()
            return True

        return self.create()

    def _create_payload_indexes_for(self, collection_name: str) -> None:
        """
        Create payload field indexes on a specific collection.

        collection_name: str — Target collection name.
        """
        keyword_fields = [
            "language",
            "file_path",
            "symbol_kind",
            "symbol_type",
            "symbol_name",
            "path_prefixes",
        ]
        for field_name in keyword_fields:
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def _migrate_legacy(self) -> None:
        """
        Migrate a legacy collection (no alias) to the versioned alias scheme.
        Renames {project} data into {project}_v1 by creating v1, copying points,
        deleting old, and creating alias.

        Qdrant has no rename API, so we create a new collection, scroll+upsert
        all points, delete the old one, then create the alias.
        """
        old_name = self._collection_name
        new_name = f"{self._collection_name}_v1"

        self._create_versioned(new_name)

        offset = None
        batch_size = 100
        while True:
            records, next_offset = self._client.scroll(
                collection_name=old_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not records:
                break

            points = []
            for record in records:
                points.append(models.PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload,
                ))

            if points:
                self._client.upsert(
                    collection_name=new_name,
                    points=points,
                    wait=True,
                )

            if next_offset is None:
                break
            offset = next_offset

        self._client.delete_collection(old_name)

        self._client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=new_name,
                        alias_name=self._collection_name,
                    )
                )
            ]
        )

    def create_shadow(self) -> str:
        """
        Create a new empty shadow collection for atomic re-indexing.
        Does NOT touch the alias — caller populates the shadow, then calls swap_alias().

        Returns: str — Name of the new shadow collection (e.g., "project_v3").
        """
        version = self._next_version()
        shadow_name = f"{self._collection_name}_v{version}"
        self._create_versioned(shadow_name)
        return shadow_name

    def copy_points(self, source_collection: str, target_collection: str, batch_size: int = 256) -> int:
        """
        Copy all points from one collection into another.

        source_collection: str — Existing collection to read from.
        target_collection: str — Empty collection to write into.
        batch_size: int — Scroll and upsert batch size.
        Returns: int — Number of copied points.
        """
        copied = 0
        offset = None

        while True:
            records, next_offset = self._client.scroll(
                collection_name=source_collection,
                limit=max(1, int(batch_size)),
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not records:
                break

            points = [
                models.PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload,
                )
                for record in records
            ]

            self._client.upsert(
                collection_name=target_collection,
                points=points,
                wait=True,
            )
            copied += len(points)

            if next_offset is None:
                break
            offset = next_offset

        return copied

    def swap_alias(self, new_collection: str) -> str | None:
        """
        Atomically swap the alias to point to a new collection.
        Returns the old collection name so the caller can delete it.

        new_collection: str — Versioned collection name to swap to.
        Returns: str | None — Previous collection name (for cleanup), or None if no prior alias.
        """
        old_collection = self.resolve_alias()

        ops: list = []
        if old_collection is not None:
            ops.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(
                        alias_name=self._collection_name,
                    )
                )
            )

        ops.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=new_collection,
                    alias_name=self._collection_name,
                )
            )
        )

        self._client.update_collection_aliases(
            change_aliases_operations=ops,
        )
        return old_collection

    def cleanup_old_versions(self, keep_current: bool = True) -> list[str]:
        """
        Delete all versioned collections except the one the alias points to.

        keep_current: bool — If True, preserve the active version. If False, delete everything.
        Returns: list[str] — Names of deleted collections.
        """
        current = self.resolve_alias() if keep_current else None
        prefix = self._collection_name + "_v"
        deleted = []

        collections = self._client.get_collections().collections
        for col in collections:
            if col.name.startswith(prefix) and col.name != current:
                suffix = col.name[len(prefix):]
                if suffix.isdigit():
                    self._client.delete_collection(col.name)
                    deleted.append(col.name)

        return deleted

    def delete(self) -> bool:
        """
        Delete the project: remove alias and all versioned collections.

        Returns: bool — True if anything was deleted, False if nothing existed.
        """
        if not self.exists():
            return False

        resolved = self.resolve_alias()
        if resolved is not None:
            self._client.update_collection_aliases(
                change_aliases_operations=[
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(
                            alias_name=self._collection_name,
                        )
                    )
                ]
            )

        self.cleanup_old_versions(keep_current=False)

        if self._collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)

        return True

    def info(self) -> dict:
        """
        Get collection info via the alias (or direct name for legacy).

        Returns: dict — Collection metadata including alias and real collection name.
        Raises: Exception — If collection doesn't exist.
        """
        real_name = self.active_collection()
        collection = self._client.get_collection(self._collection_name)
        return {
            "name": self._collection_name,
            "real_collection": real_name,
            "points_count": collection.points_count or 0,
            "indexed_vectors_count": collection.indexed_vectors_count or 0,
            "segments_count": collection.segments_count,
            "status": collection.status.value,
            "vectors": {
                name: {
                    "size": params.size,
                    "distance": params.distance.value,
                }
                for name, params in (collection.config.params.vectors or {}).items()
            },
        }

    def list_all_projects(self) -> list[dict]:
        """
        List all projects by scanning aliases. Filters out raw versioned
        collections that are backing an alias (avoids double-counting).

        Returns: list[dict] — List of project metadata dicts.
        """
        all_aliases = self._client.get_aliases()
        alias_map: dict[str, str] = {}
        backing_collections: set[str] = set()
        for alias in all_aliases.aliases:
            alias_map[alias.alias_name] = alias.collection_name
            backing_collections.add(alias.collection_name)

        result = []

        for alias_name, real_name in sorted(alias_map.items()):
            try:
                info = self._client.get_collection(alias_name)
                result.append({
                    "name": alias_name,
                    "real_collection": real_name,
                    "points_count": info.points_count or 0,
                    "status": info.status.value,
                    "vectors": {
                        name: {
                            "size": params.size,
                            "distance": params.distance.value,
                        }
                        for name, params in (info.config.params.vectors or {}).items()
                    },
                })
            except Exception:
                result.append({
                    "name": alias_name,
                    "real_collection": real_name,
                    "points_count": 0,
                    "status": "unknown",
                    "vectors": {},
                })

        collections = self._client.get_collections().collections
        for col in collections:
            if col.name in backing_collections:
                continue
            if col.name in alias_map:
                continue
            match = _VERSION_SUFFIX_RE.match(col.name)
            if match:
                continue
            try:
                info = self._client.get_collection(col.name)
                result.append({
                    "name": col.name,
                    "real_collection": col.name,
                    "points_count": info.points_count or 0,
                    "status": info.status.value,
                    "vectors": {
                        name: {
                            "size": params.size,
                            "distance": params.distance.value,
                        }
                        for name, params in (info.config.params.vectors or {}).items()
                    },
                })
            except Exception:
                result.append({
                    "name": col.name,
                    "real_collection": col.name,
                    "points_count": 0,
                    "status": "unknown",
                    "vectors": {},
                })

        return result
