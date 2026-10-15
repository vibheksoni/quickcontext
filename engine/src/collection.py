import re

from engine.src.config import EngineConfig
from engine.src.vector_store import CollectionInfo, VectorStore


_VERSION_SUFFIX_RE = re.compile(r"^(.+)_v(\d+)$")


class CollectionManager:
    """
    Manages a project's code collections with alias-based snapshot isolation.

    Collections use versioned names ({project}_v1, {project}_v2, ...) with an
    alias ({project}) pointing to the active version. Searches and reads go
    through the alias. Full re-indexes build a shadow collection and atomically
    swap the alias on completion.

    _store: VectorStore — Embedded vector store.
    _config: EngineConfig — Engine configuration with vector specs.
    _collection_name: str — Logical project name (used as alias).
    """

    def __init__(self, store: VectorStore, config: EngineConfig, collection_name: str):
        """
        store: VectorStore — Open vector store.
        config: EngineConfig — Engine config containing vector settings.
        collection_name: str — Logical project name (used as alias target).
        """
        self._store = store
        self._config = config
        self._collection_name = collection_name

    @property
    def store(self) -> VectorStore:
        """
        Returns: VectorStore — Backing store.
        """
        return self._store

    @property
    def collection_name(self) -> str:
        """
        Returns: str — Logical project name.
        """
        return self._collection_name

    def _versions(self) -> list[tuple[int, str]]:
        prefix = self._collection_name + "_v"
        versions = []
        for info in self._store.list_collections():
            suffix = info.name[len(prefix):] if info.name.startswith(prefix) else ""
            if suffix.isdigit():
                versions.append((int(suffix), info.name))
        return sorted(versions)

    def collection_exists(self, name: str) -> bool:
        """
        Return True if a real collection with this name exists.
        """
        return self._store.collection_exists(name)

    def delete_collection(self, name: str) -> bool:
        """
        Delete one real collection by name.
        """
        return self._store.delete_collection(name)

    def resolve_alias(self) -> str | None:
        """
        Returns: str | None — Real collection the project alias points to, if any.
        """
        return self._store.get_alias(self._collection_name)

    def active_collection(self) -> str:
        """
        Returns: str — Real collection backing the alias, falling back to the logical name.
        """
        return self.resolve_alias() or self._collection_name

    def exists(self) -> bool:
        """
        Returns: bool — True if the project has an alias or any versioned collection.
        """
        if self.resolve_alias() is not None:
            return True
        if self._versions():
            return True
        return self._store.collection_exists(self._collection_name)

    def _create_versioned(self, version_name: str) -> None:
        self._store.create_collection(version_name, self._config.vectors, kind="code", text_field="source")

    def create(self) -> bool:
        """
        Create {project}_v1 and point the alias at it. No-op if the project exists.

        Returns: bool — True if created.
        """
        if self.exists():
            return False
        version_name = f"{self._collection_name}_v1"
        self._create_versioned(version_name)
        self._store.set_alias(self._collection_name, version_name)
        return True

    def ensure(self) -> bool:
        """
        Ensure the project alias points at a collection, adopting the newest
        versioned collection when the alias is missing.

        Returns: bool — True if anything was created or re-pointed.
        """
        if self.resolve_alias() is not None:
            return False
        versions = self._versions()
        if versions:
            self._store.set_alias(self._collection_name, versions[-1][1])
            return True
        return self.create()

    def vectors_match_config(self, name: str | None = None) -> bool:
        """
        Return True when a collection's vector spaces match the configured ones.
        """
        info = self._store.get_collection(name or self._collection_name, with_count=False)
        if info is None:
            return True
        return tuple(info.vectors) == tuple(self._config.vectors)

    def create_shadow(self) -> str:
        """
        Create a new empty versioned collection for atomic re-indexing.
        The alias is untouched until swap_alias().

        Returns: str — Shadow collection name.
        """
        versions = self._versions()
        next_version = (versions[-1][0] + 1) if versions else 1
        shadow_name = f"{self._collection_name}_v{next_version}"
        self._create_versioned(shadow_name)
        return shadow_name

    def copy_points(self, source_collection: str, target_collection: str) -> int:
        """
        Copy all points from one collection into another.

        Returns: int — Number of copied points.
        """
        return self._store.copy_points(source_collection, target_collection)

    def swap_alias(self, new_collection: str) -> str | None:
        """
        Atomically point the alias at new_collection.

        Returns: str | None — Previous target, for cleanup.
        """
        return self._store.set_alias(self._collection_name, new_collection)

    def cleanup_old_versions(self, keep_current: bool = True) -> list[str]:
        """
        Delete versioned collections other than the active one.

        keep_current: bool — When False, delete every version.
        Returns: list[str] — Deleted collection names.
        """
        current = self.resolve_alias() if keep_current else None
        deleted = []
        for _, name in self._versions():
            if name != current and self._store.delete_collection(name):
                deleted.append(name)
        return deleted

    def delete(self) -> bool:
        """
        Delete the alias and every versioned collection for the project.

        Returns: bool — True if anything existed.
        """
        if not self.exists():
            return False
        self._store.delete_alias(self._collection_name)
        self.cleanup_old_versions(keep_current=False)
        if self._store.collection_exists(self._collection_name):
            self._store.delete_collection(self._collection_name)
        return True

    def info(self) -> dict:
        """
        Returns: dict — Metadata for the active collection.
        Raises: KeyError — If the project has no collection.
        """
        info = self._store.get_collection(self._collection_name)
        if info is None:
            raise KeyError(f"Collection not found: {self._collection_name}")
        return info.to_dict(alias=self._collection_name)

    def list_all_projects(self) -> list[dict]:
        """
        List code projects by alias, plus standalone non-versioned collections.

        Returns: list[dict] — Project metadata dicts.
        """
        return list_projects(self._store)


def list_projects(store: VectorStore, kind: str | None = "code") -> list[dict]:
    """
    List logical projects in a store: aliased collections first, then any
    standalone collection that is neither an alias target nor a stray version.
    """
    aliases = store.list_aliases()
    by_name: dict[str, CollectionInfo] = {info.name: info for info in store.list_collections()}
    backing = set(aliases.values())
    result = []

    for alias_name, real_name in sorted(aliases.items()):
        info = by_name.get(real_name)
        if info is None:
            continue
        if kind is not None and info.kind != kind:
            continue
        result.append(info.to_dict(alias=alias_name))

    for name, info in sorted(by_name.items()):
        if name in backing or name in aliases or _VERSION_SUFFIX_RE.match(name):
            continue
        if kind is not None and info.kind != kind:
            continue
        result.append(info.to_dict())

    return result


def orphaned_versions(store: VectorStore) -> list[str]:
    """
    Return versioned collections ({name}_vN) that no alias points to.
    """
    targets = set(store.list_aliases().values())
    return [
        info.name
        for info in store.list_collections()
        if info.name not in targets and _VERSION_SUFFIX_RE.match(info.name)
    ]
