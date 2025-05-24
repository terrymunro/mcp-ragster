from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection, MilvusException
import asyncio
import logging
from typing import List, Dict, Any

if __package__:
    from .config import settings
    from .exceptions import MilvusOperationError
else: 
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from ragster.config import settings
    from ragster.exceptions import MilvusOperationError

logger = logging.getLogger(__name__)

class MilvusOperator:
    def __init__(self):
        self.alias = settings.MILVUS_ALIAS
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.collection: Collection | None = None # Type hint for clarity
        # Connection and collection loading moved to an explicit connect/load method
        # to be called by the lifespan manager.

    async def connect_and_load(self):
        """Connects to Milvus and loads the collection. To be called by lifespan manager."""
        try:
            self._connect()
            self._load_collection()
            logger.info("MilvusOperator connected and collection loaded successfully.")
        except MilvusException as e:
            raise MilvusOperationError(f"Milvus client initialization or collection load failed: {e}", underlying_error=e)
        except Exception as e:
            raise MilvusOperationError(f"Unexpected error during MilvusOperator connect/load: {e}", underlying_error=e)

    def _connect(self):
        # ... (content of _connect method remains the same as in mcp_milvus_ops_final_v3)
        try:
            logger.info(f"Connecting to Milvus: host={settings.MILVUS_HOST}, port={settings.MILVUS_PORT}")
            connect_params = {"alias": self.alias, "host": settings.MILVUS_HOST, "port": settings.MILVUS_PORT}
            if settings.MILVUS_USER and settings.MILVUS_PASSWORD:
                connect_params["user"] = settings.MILVUS_USER
                connect_params["password"] = settings.MILVUS_PASSWORD
            if settings.MILVUS_USE_SSL:
                connect_params["secure"] = True
                if settings.MILVUS_SERVER_PEM_PATH: connect_params["server_pem_path"] = settings.MILVUS_SERVER_PEM_PATH
                if settings.MILVUS_SERVER_NAME: connect_params["server_name"] = settings.MILVUS_SERVER_NAME
                if settings.MILVUS_CA_CERT_PATH: connect_params["ca_certs"] = settings.MILVUS_CA_CERT_PATH
                if settings.MILVUS_CLIENT_KEY_PATH: connect_params["client_key_path"] = settings.MILVUS_CLIENT_KEY_PATH
                if settings.MILVUS_CLIENT_PEM_PATH: connect_params["client_pem_path"] = settings.MILVUS_CLIENT_PEM_PATH
                logger.info("Attempting Milvus connection with SSL enabled.")
            connections.connect(**connect_params)
            logger.info("Successfully connected to Milvus.")
        except MilvusException as e:
            logger.error(f"Failed to connect to Milvus: {e}", exc_info=True)
            raise 

    def _get_schema(self) -> CollectionSchema:
        # ... (content of _get_schema method remains the same as in mcp_milvus_ops_final_v3)
        id_field = FieldSchema(name=settings.MILVUS_ID_FIELD_NAME, dtype=DataType.INT64, is_primary=True, auto_id=True)
        embedding_field = FieldSchema(name=settings.MILVUS_INDEX_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=settings.MILVUS_VECTOR_DIMENSION)
        text_content_field = FieldSchema(name=settings.MILVUS_TEXT_FIELD_NAME, dtype=DataType.VARCHAR, max_length=65535)
        topic_field = FieldSchema(name=settings.MILVUS_TOPIC_FIELD_NAME, dtype=DataType.VARCHAR, max_length=1024)
        source_type_field = FieldSchema(name=settings.MILVUS_SOURCE_TYPE_FIELD_NAME, dtype=DataType.VARCHAR, max_length=256)
        source_identifier_field = FieldSchema(name=settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME, dtype=DataType.VARCHAR, max_length=2048)
        return CollectionSchema(
            fields=[id_field, embedding_field, text_content_field, topic_field, source_type_field, source_identifier_field],
            description="Context store for RAG application", enable_dynamic_field=False
        )

    def _load_collection(self):
        # ... (content of _load_collection method remains the same, but _create_index is called within it)
        try:
            if not utility.has_collection(self.collection_name, using=self.alias):
                logger.info(f"Collection '{self.collection_name}' does not exist. Creating...")
                self.collection = Collection(self.collection_name, schema=self._get_schema(), using=self.alias, consistency_level="Strong")
                logger.info(f"Collection '{self.collection_name}' created successfully.")
                self._create_index() 
            else:
                logger.info(f"Collection '{self.collection_name}' already exists. Loading...")
                self.collection = Collection(self.collection_name, using=self.alias)
                logger.info(f"Collection '{self.collection_name}' loaded.")
            self.collection.load()
            logger.info(f"Collection '{self.collection_name}' loaded into memory.")
        except MilvusException as e:
            logger.error(f"Error during collection load/create '{self.collection_name}': {e}", exc_info=True)
            raise

    def _create_index(self):
        # ... (content of _create_index method remains the same as in mcp_milvus_ops_final_v3)
        if not self.collection: 
            raise MilvusOperationError("Collection not initialized for index creation.")
        if any(idx.field_name == settings.MILVUS_INDEX_FIELD_NAME for idx in self.collection.indexes):
            logger.info(f"Index on field '{settings.MILVUS_INDEX_FIELD_NAME}' already exists. Skipping."); return
        logger.info(f"Creating index for field '{settings.MILVUS_INDEX_FIELD_NAME}' type '{settings.MILVUS_INDEX_TYPE}'.")
        index_params_definition = {"metric_type": settings.MILVUS_METRIC_TYPE, "index_type": settings.MILVUS_INDEX_TYPE, "params": {}}
        if settings.MILVUS_INDEX_TYPE == "HNSW":
            index_params_definition["params"] = {"M": settings.MILVUS_HNSW_M, "efConstruction": settings.MILVUS_HNSW_EF_CONSTRUCTION}
        elif settings.MILVUS_INDEX_TYPE == "IVF_FLAT":
            index_params_definition["params"] = {"nlist": settings.MILVUS_IVF_NLIST}
        try:
            self.collection.create_index(settings.MILVUS_INDEX_FIELD_NAME, index_params_definition)
            logger.info(f"Index created with params: {index_params_definition}.")
        except MilvusException as e:
            raise MilvusOperationError(f"Failed to create Milvus index: {e}", underlying_error=e)

    def _sync_insert_data(self, data_rows: List[Dict[str, Any]]) -> List[Any]:
        if not self.collection: raise MilvusOperationError("Collection not initialized for insert.")
        if not data_rows: logger.info("No data for Milvus insertion."); return []
        try:
            res = self.collection.insert(data_rows)
            self.collection.flush(); logger.info(f"Inserted {len(res.primary_keys)} entities.")
            return res.primary_keys
        except MilvusException as e:
            data_sample = str(data_rows[0])[:500] + "..." if data_rows else "N/A"
            raise MilvusOperationError(f"Failed to insert data. Sample: {data_sample}", underlying_error=e)
    
    async def insert_data(self, data_rows: List[Dict[str, Any]]) -> List[Any]:
        return await asyncio.to_thread(self._sync_insert_data, data_rows)

    def _sync_query_data(self, query_vector: List[float], top_k: int, expr: str = None) -> List[Dict[str, Any]]:
        if not self.collection: raise MilvusOperationError("Collection not initialized for query.")
        if not query_vector: raise MilvusOperationError("Query vector is empty.")
        search_params_definition = {"metric_type": settings.MILVUS_METRIC_TYPE, "params": {}}
        if settings.MILVUS_INDEX_TYPE == "HNSW":
            search_params_definition["params"] = {"ef": settings.MILVUS_SEARCH_EF}
        elif settings.MILVUS_INDEX_TYPE == "IVF_FLAT":
            search_params_definition["params"] = {"nprobe": settings.MILVUS_SEARCH_NPROBE}
        output_fields = [settings.MILVUS_ID_FIELD_NAME, settings.MILVUS_TEXT_FIELD_NAME, settings.MILVUS_TOPIC_FIELD_NAME, settings.MILVUS_SOURCE_TYPE_FIELD_NAME, settings.MILVUS_SOURCE_IDENTIFIER_FIELD_NAME]
        try:
            logger.info(f"Searching Milvus: top_k={top_k}, expr='{expr}', params={search_params_definition['params']}")
            results = self.collection.search(data=[query_vector], anns_field=settings.MILVUS_INDEX_FIELD_NAME, param=search_params_definition, limit=top_k, expr=expr, output_fields=output_fields, consistency_level="Strong")
            processed = []
            if results and results[0]:
                for hit in results[0]:
                    entity_data = {"id": hit.id, "distance": hit.distance}
                    if hit.entity:
                        for field in output_fields:
                            if field != settings.MILVUS_ID_FIELD_NAME: entity_data[field] = hit.entity.get(field)
                    processed.append(entity_data)
            logger.info(f"Milvus search found {len(processed)} results.")
            return processed
        except MilvusException as e:
            raise MilvusOperationError(f"Failed to query Milvus: {e}", underlying_error=e)
    
    async def query_data(self, query_vector: List[float], top_k: int, expr: str = None) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_query_data, query_vector, top_k, expr)

    async def close(self):
        """Placeholder for any explicit cleanup if needed, e.g., connections.disconnect(self.alias)"""
        # PyMilvus connections are typically managed globally or per alias.
        # Explicit disconnect might be useful if the alias is only for this instance.
        try:
            if connections.has_connection(self.alias):
                connections.disconnect(self.alias)
                logger.info(f"Disconnected Milvus alias '{self.alias}'.")
        except Exception as e:
            logger.error(f"Error during Milvus disconnect for alias '{self.alias}': {e}")
