"""
Configuration settings for NVIngest Powerscale Loader examples.
Edit this file to match your environment.
"""

# Nvidia NeMo Services settings
NEMO_ENDPOINT = "10.246.189.226"  # NVIDIA Ingest Rest Endpoint host
NEMO_PORT = 7670  # NVIDIA Ingest Rest port

# PowerScale connection settings
ES_HOST_URL = "https://10.246.190.192:9200"  # Elasticsearch host URL
ES_INDEX_NAME = "isi-metadataiq-index.ioflash.04bf1be5f95e87b4fa6512225f9ec530ca70"  # Elasticsearch index name
ES_API_KEY = "WnBYeGs1UUJEN2Mxc1l1WWRublo6Z1FTMEthU0JTdG1ST2xNVWw5aGliZw=="  # Elasticsearch API key (encoded)
VERIFY_SSL = False  # Verify SSL certificates for Elasticsearch

#
# File scanning settings
#
# The FOLDER_PATH must be an absolute path on the PowerScale cluster
# beginning with "/ifs"; the metadata written to ElasticSearch by
# MetadataIQ and queried by this connector uses Powerscale absolute paths
#
FOLDER_PATH = "/ifs/midx/pdfs"
FORCE_SCAN = False  # Scan all files, ignoring checkpoint

# Debug settings
DEBUG_MODE = False  # Set to True for more verbose logging
