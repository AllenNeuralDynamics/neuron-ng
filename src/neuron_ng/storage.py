"""Upload written layers to S3.

Layers are written to a local directory first, then uploaded as is.
Neuroglancer loads them from the browser, so the bucket must be publicly
readable, with CORS that allows ``GET``. Needs the ``s3`` extra (``boto3``)
unless a client is passed.
"""
import logging
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

#: Precomputed ``info`` files have no extension but are JSON.
_INFO_CONTENT_TYPE = "application/json"
_DEFAULT_CONTENT_TYPE = "application/octet-stream"


def parse_s3_uri(uri):
    """Split ``s3://bucket/key/prefix`` into ``("bucket", "key/prefix")``."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3:// URI: {uri!r}")
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    if not bucket:
        raise ValueError(f"No bucket in S3 URI: {uri!r}")
    return bucket, prefix.strip("/")


def content_type(path):
    path = Path(path)
    if path.name == "info":
        return _INFO_CONTENT_TYPE
    return mimetypes.guess_type(path.name)[0] or _DEFAULT_CONTENT_TYPE


def upload_dir(local_dir, s3_uri, client=None, max_workers=16):
    """Upload every file under ``local_dir`` to ``s3_uri``, keeping relative paths.

    Returns the number of files uploaded. Precomputed layers are many small
    files, so uploads run in parallel.
    """
    if client is None:
        import boto3

        client = boto3.client("s3")
    bucket, prefix = parse_s3_uri(s3_uri)
    local_dir = Path(local_dir)
    files = sorted(p for p in local_dir.rglob("*") if p.is_file())

    def put(path):
        key = "/".join(filter(None, [prefix, path.relative_to(local_dir).as_posix()]))
        client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type(path)})

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # list() re-raises the first upload error.
        list(pool.map(put, files))
    logger.info("Uploaded %d file(s) to %s", len(files), s3_uri)
    return len(files)
