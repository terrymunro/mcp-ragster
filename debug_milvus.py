#!/usr/bin/env python3
"""Debug script to inspect Milvus collection data."""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ragster.milvus_ops import MilvusOperator
from ragster.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_milvus():
    """Debug Milvus collection structure and data."""
    try:
        milvus_op = MilvusOperator()
        
        # Get collection info
        logger.info("=== Collection Info ===")
        stats = milvus_op.client.get_collection_stats(milvus_op.collection_name)
        logger.info(f"Collection stats: {stats}")
        
        # Try to describe the collection
        try:
            desc = milvus_op.client.describe_collection(milvus_op.collection_name)
            logger.info(f"Collection description: {desc}")
        except Exception as e:
            logger.warning(f"Could not describe collection: {e}")
        
        # Try to get all fields with different query
        logger.info("=== Trying query with all fields ===")
        try:
            results = milvus_op.client.query(
                collection_name=milvus_op.collection_name,
                expr="",
                output_fields=["*"],
                limit=1
            )
            logger.info(f"Query with all fields result: {results}")
            if results:
                logger.info(f"Available fields: {list(results[0].keys())}")
        except Exception as e:
            logger.warning(f"Query with all fields failed: {e}")
        
        # Try to get specific fields
        logger.info("=== Trying query with specific fields ===")
        try:
            results = milvus_op.client.query(
                collection_name=milvus_op.collection_name,
                expr="",
                output_fields=[
                    settings.MILVUS_TEXT_FIELD_NAME,
                    settings.MILVUS_TOPIC_FIELD_NAME,
                ],
                limit=1
            )
            logger.info(f"Query with specific fields result: {results}")
        except Exception as e:
            logger.warning(f"Query with specific fields failed: {e}")
            
        # Close the client
        milvus_op.close()
        
    except Exception as e:
        logger.error(f"Error during debug: {e}", exc_info=True)

if __name__ == "__main__":
    debug_milvus() 