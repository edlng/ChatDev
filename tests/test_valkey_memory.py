"""Tests for Valkey memory store: config validation, registry, and integration skeleton."""

from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from entity.configs.base import ConfigError
from entity.configs.node.memory import ValkeyMemoryConfig, EmbeddingConfig


# =============================================================================
# Unit Tests: ValkeyMemoryConfig
# =============================================================================


class TestValkeyMemoryConfigFromDict:
    """Test ValkeyMemoryConfig.from_dict parsing and validation."""

    def test_minimal_config(self):
        """Parses with all defaults when only empty dict provided."""
        cfg = ValkeyMemoryConfig.from_dict({}, path="test")
        assert cfg.host == "localhost"
        assert cfg.port == 6379
        assert cfg.password is None
        assert cfg.db == 0
        assert cfg.index_name == "memory_index"
        assert cfg.key_prefix == "memory:"
        assert cfg.ttl_seconds is None
        assert cfg.embedding is None

    def test_full_config(self):
        """Parses all fields correctly."""
        data = {
            "host": "valkey.internal",
            "port": 6380,
            "password": "secret",
            "db": 2,
            "index_name": "chatdev_memory",
            "key_prefix": "agent:",
            "ttl_seconds": 86400,
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
            },
        }
        cfg = ValkeyMemoryConfig.from_dict(data, path="test")
        assert cfg.host == "valkey.internal"
        assert cfg.port == 6380
        assert cfg.password == "secret"
        assert cfg.db == 2
        assert cfg.index_name == "chatdev_memory"
        assert cfg.key_prefix == "agent:"
        assert cfg.ttl_seconds == 86400
        assert cfg.embedding is not None
        assert cfg.embedding.provider == "openai"
        assert cfg.embedding.model == "text-embedding-3-small"

    def test_invalid_port_zero(self):
        """Port must be at least 1."""
        with pytest.raises(ConfigError, match="port"):
            ValkeyMemoryConfig.from_dict({"port": 0}, path="test")

    def test_invalid_port_negative(self):
        """Negative port rejected."""
        with pytest.raises(ConfigError, match="port"):
            ValkeyMemoryConfig.from_dict({"port": -1}, path="test")

    def test_invalid_port_string(self):
        """Non-integer port rejected."""
        with pytest.raises(ConfigError, match="port"):
            ValkeyMemoryConfig.from_dict({"port": "6379"}, path="test")

    def test_invalid_port_too_high(self):
        """Port above 65535 rejected."""
        with pytest.raises(ConfigError, match="port"):
            ValkeyMemoryConfig.from_dict({"port": 65536}, path="test")

    def test_invalid_db_negative(self):
        """Negative db index rejected."""
        with pytest.raises(ConfigError, match="db"):
            ValkeyMemoryConfig.from_dict({"db": -1}, path="test")

    def test_invalid_db_too_high(self):
        """Database index above 15 rejected."""
        with pytest.raises(ConfigError, match="db"):
            ValkeyMemoryConfig.from_dict({"db": 16}, path="test")

    def test_invalid_ttl_zero(self):
        """TTL must be positive if specified."""
        with pytest.raises(ConfigError, match="ttl_seconds"):
            ValkeyMemoryConfig.from_dict({"ttl_seconds": 0}, path="test")

    def test_invalid_ttl_negative(self):
        """Negative TTL rejected."""
        with pytest.raises(ConfigError, match="ttl_seconds"):
            ValkeyMemoryConfig.from_dict({"ttl_seconds": -100}, path="test")

    def test_invalid_ttl_string(self):
        """Non-integer TTL rejected."""
        with pytest.raises(ConfigError, match="ttl_seconds"):
            ValkeyMemoryConfig.from_dict({"ttl_seconds": "3600"}, path="test")

    def test_optional_fields_none_when_omitted(self):
        """TTL and embedding are None when not specified."""
        cfg = ValkeyMemoryConfig.from_dict({"host": "localhost"}, path="test")
        assert cfg.ttl_seconds is None
        assert cfg.embedding is None

    def test_embedding_none_when_explicitly_null(self):
        """Embedding config is None when explicitly set to null."""
        cfg = ValkeyMemoryConfig.from_dict({"embedding": None}, path="test")
        assert cfg.embedding is None


class TestValkeyMemoryConfigFieldSpecs:
    """Test FIELD_SPECS metadata for UI generation."""

    def test_field_specs_defined(self):
        """All expected fields have specs."""
        specs = ValkeyMemoryConfig.field_specs()
        expected_fields = {"host", "port", "password", "db", "index_name", "key_prefix", "ttl_seconds", "embedding"}
        assert expected_fields.issubset(set(specs.keys()))

    def test_host_not_required(self):
        """Host has a default so it's not required."""
        specs = ValkeyMemoryConfig.field_specs()
        assert specs["host"].required is False

    def test_index_name_not_required(self):
        """index_name has a default."""
        specs = ValkeyMemoryConfig.field_specs()
        assert specs["index_name"].required is False
        assert specs["index_name"].default == "memory_index"

    def test_ttl_not_required(self):
        """ttl_seconds is optional."""
        specs = ValkeyMemoryConfig.field_specs()
        assert specs["ttl_seconds"].required is False

    def test_embedding_has_child(self):
        """Embedding spec references EmbeddingConfig as child."""
        specs = ValkeyMemoryConfig.field_specs()
        assert specs["embedding"].child is EmbeddingConfig


# =============================================================================
# Unit Tests: Registry
# =============================================================================


class TestValkeyRegistration:
    """Test that 'valkey' is properly registered as a memory store type."""

    def test_valkey_in_registry(self):
        """'valkey' appears in memory store registrations."""
        from runtime.node.agent.memory.registry import iter_memory_store_registrations

        stores = iter_memory_store_registrations()
        assert "valkey" in stores

    def test_valkey_config_cls(self):
        """Registry entry points to ValkeyMemoryConfig."""
        from runtime.node.agent.memory.registry import get_memory_store_registration

        reg = get_memory_store_registration("valkey")
        assert reg.config_cls is ValkeyMemoryConfig

    def test_valkey_has_summary(self):
        """Registry entry has a non-empty summary."""
        from runtime.node.agent.memory.registry import get_memory_store_registration

        reg = get_memory_store_registration("valkey")
        assert reg.summary and len(reg.summary) > 0

    def test_schema_registry_has_valkey(self):
        """schema_registry also knows about 'valkey' for config parsing."""
        from schema_registry import get_memory_store_schema

        schema = get_memory_store_schema("valkey")
        assert schema.config_cls is ValkeyMemoryConfig


# =============================================================================
# Unit Tests: MemoryStoreConfig can parse type=valkey
# =============================================================================


class TestMemoryStoreConfigValkey:
    """Test that MemoryStoreConfig.from_dict handles type='valkey'."""

    def test_parses_valkey_store(self):
        """Full memory store config with type=valkey parses successfully."""
        from entity.configs.node.memory import MemoryStoreConfig

        data = {
            "name": "chatdev_memory",
            "type": "valkey",
            "config": {
                "host": "localhost",
                "port": 6379,
                "index_name": "chatdev_memory",
                "ttl_seconds": 86400,
            },
        }
        store = MemoryStoreConfig.from_dict(data, path="test")
        assert store.name == "chatdev_memory"
        assert store.type == "valkey"
        assert isinstance(store.config, ValkeyMemoryConfig)
        assert store.config.index_name == "chatdev_memory"
        assert store.config.ttl_seconds == 86400

    def test_rejects_missing_config_block(self):
        """Config block is required."""
        from entity.configs.node.memory import MemoryStoreConfig

        data = {
            "name": "bad",
            "type": "valkey",
            "config": None,
        }
        with pytest.raises(ConfigError):
            MemoryStoreConfig.from_dict(data, path="test")


# =============================================================================
# Integration Test Skeleton: ValkeyMemory (requires running Valkey server)
# =============================================================================


@pytest.mark.skipif(True, reason="Integration test — requires running Valkey server with Search module")
class TestValkeyMemoryIntegration:
    """
    Integration tests for ValkeyMemory.

    Prerequisites:
    - Valkey server running on localhost:6379
    - Valkey Search module loaded (valkey-server --loadmodule valkeysearch.so)
    - Or use the valkey/valkey-bundle Docker image which includes Search:
      docker run -p 6379:6379 valkey/valkey-bundle:latest
    - valkey-glide installed (pip install .[valkey])

    Run with: pytest tests/test_valkey_memory.py::TestValkeyMemoryIntegration -v --no-header
    """

    @pytest.fixture
    def valkey_store(self):
        """Create a MemoryStoreConfig for integration testing."""
        from entity.configs.node.memory import MemoryStoreConfig

        data = {
            "name": "integration_test",
            "type": "valkey",
            "config": {
                "host": "localhost",
                "port": 6379,
                "index_name": "test_memory_idx",
                "key_prefix": "test:memory:",
                "ttl_seconds": 60,
                "embedding": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "api_key": "test-key",
                },
            },
        }
        return MemoryStoreConfig.from_dict(data, path="integration")

    @pytest.fixture
    def memory(self, valkey_store):
        """Create a ValkeyMemory instance and clean up after test."""
        from runtime.node.agent.memory.valkey_memory import ValkeyMemory

        mem = ValkeyMemory(valkey_store)
        yield mem
        # Cleanup: drop index and keys
        # mem._cleanup_test_data()

    def test_load_is_noop(self, memory):
        """load() completes without error (server handles persistence)."""
        memory.load()

    def test_save_is_noop(self, memory):
        """save() completes without error (server handles persistence)."""
        memory.save()

    def test_update_stores_memory(self, memory):
        """update() writes a memory item to Valkey as a hash."""
        from runtime.node.agent.memory.memory_base import (
            MemoryContentSnapshot,
            MemoryWritePayload,
        )

        payload = MemoryWritePayload(
            agent_role="writer",
            inputs_text="The capital of France is Paris",
            input_snapshot=MemoryContentSnapshot(text="The capital of France is Paris"),
            output_snapshot=MemoryContentSnapshot(text="Noted."),
        )
        memory.update(payload)
        # Verify key exists in Valkey
        # assert memory.count_memories() >= 1

    def test_retrieve_returns_relevant_items(self, memory):
        """retrieve() finds stored memories via KNN search."""
        from runtime.node.agent.memory.memory_base import (
            MemoryContentSnapshot,
            MemoryWritePayload,
        )

        # Store a fact
        payload = MemoryWritePayload(
            agent_role="writer",
            inputs_text="Python was created by Guido van Rossum",
            input_snapshot=MemoryContentSnapshot(text="Python was created by Guido van Rossum"),
            output_snapshot=None,
        )
        memory.update(payload)

        # Query for it
        query = MemoryContentSnapshot(text="Who created Python?")
        results = memory.retrieve("writer", query, top_k=3, similarity_threshold=-1.0)

        assert len(results) >= 1
        assert "Python" in results[0].content_summary or "Guido" in results[0].content_summary

    def test_retrieve_empty_query_returns_empty(self, memory):
        """Empty query returns empty list without searching."""
        from runtime.node.agent.memory.memory_base import MemoryContentSnapshot

        query = MemoryContentSnapshot(text="   ")
        results = memory.retrieve("writer", query, top_k=3, similarity_threshold=-1.0)
        assert results == []

    def test_retrieve_respects_top_k(self, memory):
        """Number of results does not exceed top_k."""
        from runtime.node.agent.memory.memory_base import (
            MemoryContentSnapshot,
            MemoryWritePayload,
        )

        # Store multiple items
        for i in range(5):
            payload = MemoryWritePayload(
                agent_role="writer",
                inputs_text=f"Fact number {i} about testing",
                input_snapshot=MemoryContentSnapshot(text=f"Fact number {i} about testing"),
                output_snapshot=None,
            )
            memory.update(payload)

        query = MemoryContentSnapshot(text="testing facts")
        results = memory.retrieve("writer", query, top_k=2, similarity_threshold=-1.0)
        assert len(results) <= 2

    def test_ttl_expiry(self, memory):
        """Items expire after ttl_seconds (would need time manipulation or short TTL)."""
        # This test would require either:
        # 1. Setting ttl_seconds=1 and sleeping
        # 2. Using Valkey's DEBUG SLEEP or TIME commands
        pass

    def test_index_created_lazily(self, memory):
        """Index is created on first use, not at construction time."""
        # Verify FT.INFO raises (no index) before first operation
        # Then after first update, FT.INFO succeeds
        pass

    def test_import_error_without_glide(self):
        """Clear ImportError when valkey-glide is not installed."""
        with patch.dict("sys.modules", {"glide": None}):
            with pytest.raises(ImportError, match="valkey-glide"):
                from runtime.node.agent.memory import valkey_memory  # noqa: F401
                # Force re-import
                import importlib
                importlib.reload(valkey_memory)
