import sys
from pathlib import Path
import types

import pytest

# Ensure src is on the path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class MockMilvusModule(types.ModuleType):
    class MilvusException(Exception):
        pass

    class DataType:
        INT64 = "INT64"
        FLOAT_VECTOR = "FLOAT_VECTOR"
        VARCHAR = "VARCHAR"

    class FieldSchema:
        def __init__(self, name: str, dtype: str, **_kwargs):
            self.name = name
            self.dtype = dtype

    class CollectionSchema:
        def __init__(
            self,
            fields,
            description: str | None = None,
            enable_dynamic_field: bool = False,
        ):
            self.fields = fields
            self.description = description
            self.enable_dynamic_field = enable_dynamic_field

    class Connections:
        def connect(self, **_kwargs):
            pass

        def has_connection(self, _alias: str) -> bool:
            return True

        def disconnect(self, _alias: str):
            pass

    class Utility:
        def has_collection(self, _name: str, using: str | None = None) -> bool:
            return True

    class Collection:
        def __init__(
            self,
            name: str,
            schema=None,
            using: str | None = None,
            consistency_level: str | None = None,
        ):
            self.name = name
            self.schema = schema
            self.using = using
            self.consistency_level = consistency_level
            self.indexes = []
            self.inserted = []
            self.flushed = False
            self.create_index_called = 0
            self.search_called = False

        def create_index(self, field_name: str, params):
            self.create_index_called += 1
            self.indexes.append(
                types.SimpleNamespace(field_name=field_name, params=params)
            )

        def insert(self, data):
            self.inserted.append(data)
            return types.SimpleNamespace(primary_keys=list(range(1, len(data) + 1)))

        def flush(self):
            self.flushed = True

        def search(
            self, data, anns_field, param, limit, expr, output_fields, consistency_level
        ):
            self.search_called = True
            entity = {field: f"value_{field}" for field in output_fields}
            hit = types.SimpleNamespace(id=1, distance=0.5, entity=entity)
            return [[hit]]

    def __init__(self):
        super().__init__("pymilvus")
        self.connections = self.Connections()
        self.utility = self.Utility()
        self.Collection = self.Collection
        self.CollectionSchema = self.CollectionSchema
        self.DataType = self.DataType
        self.FieldSchema = self.FieldSchema
        self.MilvusException = self.MilvusException


@pytest.fixture
def mock_pymilvus(monkeypatch):
    module = MockMilvusModule()
    monkeypatch.setitem(sys.modules, "pymilvus", module)
    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_mod)
    yield module
    monkeypatch.delitem(sys.modules, "pymilvus")
    monkeypatch.delitem(sys.modules, "dotenv")


def test_get_schema_field_names(mock_pymilvus):
    from ragster.milvus_ops import MilvusOperator
    from ragster.config import settings

    op = MilvusOperator()
    schema = op._get_schema()
    names = [f.name for f in schema.fields]

    assert settings.MILVUS_ID_FIELD_NAME in names
    assert settings.MILVUS_INDEX_FIELD_NAME in names
    assert settings.MILVUS_TEXT_FIELD_NAME in names
    assert settings.MILVUS_TOPIC_FIELD_NAME in names
    assert settings.MILVUS_SOURCE_TYPE_FIELD_NAME in names
    assert settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME in names


def test_create_index_skips_when_exists(mock_pymilvus):
    from ragster.milvus_ops import MilvusOperator
    from ragster.config import settings

    op = MilvusOperator()
    col = mock_pymilvus.Collection("c")
    col.indexes.append(
        types.SimpleNamespace(field_name=settings.MILVUS_INDEX_FIELD_NAME)
    )
    op.collection = col

    op._create_index()

    assert col.create_index_called == 0
    assert len(col.indexes) == 1


def test_sync_insert_data_success(mock_pymilvus):
    from ragster.milvus_ops import MilvusOperator

    op = MilvusOperator()
    col = mock_pymilvus.Collection("c")
    op.collection = col

    rows = [{"text": "a"}, {"text": "b"}]
    ids = op._sync_insert_data(rows)

    assert ids == [1, 2]
    assert col.inserted == [rows]
    assert col.flushed


def test_sync_insert_data_error(mock_pymilvus):
    from ragster.milvus_ops import MilvusOperator, MilvusOperationError

    op = MilvusOperator()
    col = mock_pymilvus.Collection("c")

    def raise_exc(_data):
        raise mock_pymilvus.MilvusException("boom")

    col.insert = raise_exc
    op.collection = col

    with pytest.raises(MilvusOperationError):
        op._sync_insert_data([{"text": "a"}])


def test_sync_query_data_success(mock_pymilvus):
    from ragster.milvus_ops import MilvusOperator

    op = MilvusOperator()
    col = mock_pymilvus.Collection("c")
    op.collection = col

    results = op._sync_query_data([0.1], top_k=1)
    assert results
    assert results[0]["id"] == 1
    assert "distance" in results[0]


def test_sync_query_data_error(mock_pymilvus):
    from ragster.milvus_ops import MilvusOperator, MilvusOperationError

    op = MilvusOperator()
    col = mock_pymilvus.Collection("c")

    def raise_exc(*_args, **_kwargs):
        raise mock_pymilvus.MilvusException("fail")

    col.search = raise_exc
    op.collection = col

    with pytest.raises(MilvusOperationError):
        op._sync_query_data([0.1], top_k=1)
