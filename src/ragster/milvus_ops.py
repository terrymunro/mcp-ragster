import logging
from typing import Any, cast

from pymilvus import (
    MilvusClient,
    MilvusException,
    DataType,
)

from .config import settings
from .exceptions import MilvusOperationError

logger = logging.getLogger(__name__)


class MilvusOperator:
    collection_name: str
    client: MilvusClient

    def __init__(self):
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        # Create client with optional token - ensure all params are keyword args
        try:
            if settings.MILVUS_TOKEN:
                self.client = MilvusClient(
                    uri=settings.MILVUS_URI,
                    db_name=settings.MILVUS_DB,
                    token=settings.MILVUS_TOKEN,
                )
            else:
                self.client = MilvusClient(
                    uri=settings.MILVUS_URI, db_name=settings.MILVUS_DB
                )
        except Exception as e:
            logger.error(f"Failed to initialize MilvusClient: {e}")
            raise

    def load_collection(self) -> None:
        """Loads the collection. To be called by lifespan manager."""
        try:
            if not self.client.has_collection(self.collection_name):
                logger.info(
                    f"Collection '{self.collection_name}' does not exist. Creating..."
                )
                self._create_collection_with_schema()
                logger.info(
                    f"Collection '{self.collection_name}' created successfully."
                )
                self._create_index()
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
            self.client.load_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' loaded into memory.")
        except MilvusException as e:
            logger.error(
                f"Error during collection load/create '{self.collection_name}': {e}",
                exc_info=True,
            )
            raise

    def _create_collection_with_schema(self):
        """Create collection using proper schema format"""
        # Create schema using MilvusClient.create_schema()
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )

        # Add fields to schema
        schema.add_field(
            field_name=settings.MILVUS_ID_FIELD_NAME,
            datatype=DataType.INT64,
            is_primary=True,
        )

        schema.add_field(
            field_name=settings.MILVUS_INDEX_FIELD_NAME,
            datatype=DataType.FLOAT_VECTOR,
            dim=settings.MILVUS_VECTOR_DIMENSION,
        )

        schema.add_field(
            field_name=settings.MILVUS_TEXT_FIELD_NAME,
            datatype=DataType.VARCHAR,
            max_length=65535,
        )

        schema.add_field(
            field_name=settings.MILVUS_TOPIC_FIELD_NAME,
            datatype=DataType.VARCHAR,
            max_length=1024,
        )

        schema.add_field(
            field_name=settings.MILVUS_SOURCE_TYPE_FIELD_NAME,
            datatype=DataType.VARCHAR,
            max_length=256,
        )

        schema.add_field(
            field_name=settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME,
            datatype=DataType.VARCHAR,
            max_length=2048,
        )

        # Create collection with schema
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            consistency_level="Strong",
        )

    def _create_index(self):
        # Check if index exists
        indexes = self.client.list_indexes(self.collection_name)
        if any(
            idx["field_name"] == settings.MILVUS_INDEX_FIELD_NAME for idx in indexes
        ):
            logger.info(
                f"Index on field '{settings.MILVUS_INDEX_FIELD_NAME}' already exists. Skipping."
            )
            return
        logger.info(
            f"Creating index for field '{settings.MILVUS_INDEX_FIELD_NAME}' type '{settings.MILVUS_INDEX_TYPE}'."
        )

        # Prepare index parameters
        index_params = self.client.prepare_index_params()

        # Add index for vector field
        if settings.MILVUS_INDEX_TYPE == "HNSW":
            index_params.add_index(
                field_name=settings.MILVUS_INDEX_FIELD_NAME,
                index_type=settings.MILVUS_INDEX_TYPE,
                metric_type=settings.MILVUS_METRIC_TYPE,
                params={
                    "M": settings.MILVUS_HNSW_M,
                    "efConstruction": settings.MILVUS_HNSW_EF_CONSTRUCTION,
                },
            )
        elif settings.MILVUS_INDEX_TYPE == "IVF_FLAT":
            index_params.add_index(
                field_name=settings.MILVUS_INDEX_FIELD_NAME,
                index_type=settings.MILVUS_INDEX_TYPE,
                metric_type=settings.MILVUS_METRIC_TYPE,
                params={"nlist": settings.MILVUS_IVF_NLIST},
            )
        else:
            # Use AUTOINDEX as fallback
            index_params.add_index(
                field_name=settings.MILVUS_INDEX_FIELD_NAME,
                index_type="AUTOINDEX",
                metric_type=settings.MILVUS_METRIC_TYPE,
            )

        try:
            self.client.create_index(
                collection_name=self.collection_name, index_params=index_params
            )
            logger.info(
                f"Index created successfully for field '{settings.MILVUS_INDEX_FIELD_NAME}'."
            )
        except MilvusException as e:
            raise MilvusOperationError(
                f"Failed to create Milvus index: {e}", underlying_error=e
            )

    def insert_data(self, data_rows: list[dict[str, Any]]) -> list[Any]:
        if not self.client:
            raise MilvusOperationError("Milvus client not initialized for insert.")
        if not data_rows:
            logger.info("No data for Milvus insertion.")
            return []
        try:
            res = self.client.insert(
                collection_name=self.collection_name, data=data_rows
            )
            # Handle Milvus response format: {'insert_count': 1, 'ids': [...], 'cost': 2}
            if isinstance(res, dict):
                ids = res.get("ids", [])
                insert_count = res.get("insert_count", len(ids))
                logger.info(f"Inserted {insert_count} entities.")
                return list(ids)  # Convert to regular list
            else:
                logger.info(f"Insert completed with unexpected response: {type(res)}")
                return []
        except MilvusException as e:
            data_sample = str(data_rows[0])[:500] + "..." if data_rows else "N/A"
            raise MilvusOperationError(
                f"Failed to insert data. Sample: {data_sample}", underlying_error=e
            )

    def query_data(
        self,
        query_vector: list[float],
        top_k: int,
        expr: str | None = None,
        search_ef: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.client:
            raise MilvusOperationError("Milvus client not initialized for query.")
        if not query_vector:
            raise MilvusOperationError("Query vector is empty.")

        # Prepare search parameters
        search_params = {}
        if settings.MILVUS_INDEX_TYPE == "HNSW":
            ef_value = search_ef if search_ef is not None else settings.MILVUS_SEARCH_EF
            search_params = {"ef": ef_value}
        elif settings.MILVUS_INDEX_TYPE == "IVF_FLAT":
            search_params = {"nprobe": settings.MILVUS_SEARCH_NPROBE}

        output_fields = [
            settings.MILVUS_TEXT_FIELD_NAME,
            settings.MILVUS_TOPIC_FIELD_NAME,
            settings.MILVUS_SOURCE_TYPE_FIELD_NAME,
            settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME,
        ]

        try:
            logger.info(
                f"Searching Milvus: top_k={top_k}, expr='{expr}', params={search_params}"
            )
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field=settings.MILVUS_INDEX_FIELD_NAME,
                search_params=search_params,
                limit=top_k,
                output_fields=output_fields,
                consistency_level="Strong",
            )
            results_seq = cast(
              
          [Any], results)
            processed = []
            try:
                # Handle different result types from Milvus search
                if results_seq:
                    # Try to access the first result set
                    first_result = None
                    try:
                        first_result = (
                            cast(list[Any], results)[0]
                            if hasattr(results, "__getitem__")
                            else None
                        )
                    except (IndexError, TypeError):
                        first_result = None

                    if first_result:
                        for hit in first_result:
                            entity_data = {"id": hit.id, "distance": hit.distance}
                            if hit.entity:
                                for field in output_fields:
                                    if field != settings.MILVUS_ID_FIELD_NAME:
                                        entity_data[field] = hit.entity.get(field)
                            processed.append(entity_data)
            except (AttributeError, IndexError, TypeError) as e:
                logger.warning(f"Error processing Milvus search results: {e}")
                processed = []
            logger.info(f"Milvus search found {len(processed)} results.")
            return processed
        except MilvusException as e:
            raise MilvusOperationError(
                f"Failed to query Milvus: {e}", underlying_error=e
            )

    def health_check(self) -> bool:
        try:
            if not self.client:
                return False
            # Try a simple operation to verify connection
            stats = self.client.get_collection_stats(self.collection_name)
            return bool(stats and stats.get("row_count", 0) >= 0)
        except Exception as e:
            logger.warning(f"Milvus health check failed: {e}")
            return False

    def reconnect(self) -> bool:
        try:
            # Reinitialize client with optional token
            if settings.MILVUS_TOKEN:
                self.client = MilvusClient(
                    uri=settings.MILVUS_URI,
                    db_name=settings.MILVUS_DB,
                    token=settings.MILVUS_TOKEN,
                )
            else:
                self.client = MilvusClient(
                    uri=settings.MILVUS_URI, db_name=settings.MILVUS_DB
                )
            self.load_collection()
            logger.info(
                f"Milvus reconnected and collection reloaded for '{self.collection_name}'"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to reconnect Milvus: {e}")
            return False

    def get_stored_topics(self, limit: int = 10) -> list[str]:
        try:
            if not self.client:
                return []
            results = self.client.query(
                collection_name=self.collection_name,
                expr="",
                output_fields=[settings.MILVUS_TOPIC_FIELD_NAME],
                limit=limit * 3,
            )
            topics = list(
                set(result[settings.MILVUS_TOPIC_FIELD_NAME] for result in results)
            )
            return topics[:limit]
        except Exception as e:
            logger.warning(f"Failed to get stored topics for warm-up: {e}")
            return []

    def has_data(self) -> bool:
        try:
            if not self.client:
                return False
            stats = self.client.get_collection_stats(self.collection_name)
            return bool(stats and stats.get("row_count", 0) > 0)
        except Exception:
            return False

    def close(self):
        self.client.close()
