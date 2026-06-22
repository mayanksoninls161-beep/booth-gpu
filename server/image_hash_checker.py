# # # import os
# # # import json
# # # import io
# # # from datetime import datetime, timezone
# # # from PIL import Image
# # # import imagehash
# # # import piexif

# # # # ─────────────────────────────────────────────────────────────────
# # # # CONFIG — persistent path
# # # # ─────────────────────────────────────────────────────────────────
# # # # Put hash_db.json OUTSIDE the deployment folder so a fresh git deploy
# # # # doesn't wipe it out. Set HASH_DB_PATH env var on your server to
# # # # something persistent, e.g. /var/data/hash_db.json on Render,
# # # # /opt/persistent/hash_db.json on a VPS, or a mounted disk path.
# # # #
# # # # Default falls back to a folder called "data/" one level above the
# # # # project, which usually survives deploys; override in production.
# # # HASH_DB_PATH = os.getenv(
# # #     "HASH_DB_PATH",
# # #     os.path.join(
# # #         os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
# # #         "data",
# # #         "hash_db.json",
# # #     ),
# # # )

# # # print(HASH_DB_PATH)

# # # # Make sure parent directory exists
# # # os.makedirs(os.path.dirname(HASH_DB_PATH), exist_ok=True)


# # # # ─────────────────────────────────────────────────────────────────
# # # # LOAD / SAVE
# # # # ─────────────────────────────────────────────────────────────────
# # # def load_hash_db():
# # #     if not os.path.exists(HASH_DB_PATH):
# # #         return set()

# # #     try:
# # #         with open(HASH_DB_PATH, "r") as f:
# # #             content = f.read().strip()
# # #             if not content:
# # #                 return set()
# # #             return set(json.loads(content))
# # #     except json.JSONDecodeError:
# # #         print("WARNING: hash_db.json is corrupted. Resetting.")
# # #         return set()


# # # def save_hash_db(hash_set):
# # #     # Write atomically: write to temp file then rename, so a crash
# # #     # mid-write doesn't corrupt the DB.
# # #     tmp_path = HASH_DB_PATH + ".tmp"
# # #     with open(tmp_path, "w") as f:
# # #         json.dump(list(hash_set), f, indent=2)
# # #     os.replace(tmp_path, HASH_DB_PATH)


# # # # ─────────────────────────────────────────────────────────────────
# # # # IMAGE METADATA (EXIF) — embed hash + date
# # # # ─────────────────────────────────────────────────────────────────
# # # # We store our data inside EXIF UserComment as a small JSON blob:
# # # #   {"phash": "abc123...", "date": "2026-04-27T12:34:56Z"}

# # # _USER_COMMENT_PREFIX = b"ASCII\x00\x00\x00"  # standard EXIF UserComment charset prefix


# # # def _parse_user_comment(raw):
# # #     if not raw:
# # #         return None
# # #     payload = raw[8:] if len(raw) > 8 else raw
# # #     try:
# # #         text = payload.decode("utf-8", errors="ignore").strip("\x00").strip()
# # #         if not text:
# # #             return None
# # #         return json.loads(text)
# # #     except (json.JSONDecodeError, UnicodeDecodeError):
# # #         return None


# # # def read_image_metadata(raw_bytes: bytes):
# # #     """
# # #     Read embedded {phash, date} from image EXIF UserComment.
# # #     Returns (phash, date_str) or (None, None) if absent / unsupported format.
# # #     """
# # #     try:
# # #         exif_dict = piexif.load(raw_bytes)
# # #     except Exception:
# # #         return None, None

# # #     user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
# # #     parsed = _parse_user_comment(user_comment)
# # #     if not parsed:
# # #         return None, None

# # #     return parsed.get("phash"), parsed.get("date")


# # # def write_image_metadata(raw_bytes: bytes, phash: str, date_str: str = None):
# # #     """
# # #     Embed {phash, date} into image EXIF UserComment.
# # #     If both are ALREADY present (and phash matches), returns the original bytes
# # #     unchanged. Otherwise returns new bytes with the metadata written.

# # #     Only works reliably for JPEG. Non-JPEG returns unchanged bytes.
# # #     """
# # #     if date_str is None:
# # #         date_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# # #     try:
# # #         img = Image.open(io.BytesIO(raw_bytes))
# # #         fmt = (img.format or "").upper()
# # #     except Exception:
# # #         return raw_bytes

# # #     if fmt != "JPEG":
# # #         return raw_bytes

# # #     # Already tagged with the same hash? leave it alone
# # #     existing_hash, existing_date = read_image_metadata(raw_bytes)
# # #     if existing_hash == phash and existing_date:
# # #         return raw_bytes

# # #     try:
# # #         exif_dict = piexif.load(raw_bytes)
# # #     except Exception:
# # #         exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

# # #     payload = json.dumps({"phash": phash, "date": date_str}).encode("utf-8")
# # #     exif_dict.setdefault("Exif", {})[piexif.ExifIFD.UserComment] = (
# # #         _USER_COMMENT_PREFIX + payload
# # #     )

# # #     try:
# # #         exif_bytes = piexif.dump(exif_dict)
# # #     except Exception:
# # #         return raw_bytes

# # #     out = io.BytesIO()
# # #     img.save(out, format="JPEG", exif=exif_bytes, quality=95)
# # #     return out.getvalue()

# # import os
# # import json
# # import io
# # import re
# # from datetime import datetime, timezone
# # from urllib.parse import urlparse
# # from PIL import Image
# # import imagehash
# # import piexif

# # # ─────────────────────────────────────────────────────────────────
# # # CONFIG — persistent path
# # # ─────────────────────────────────────────────────────────────────
# # HASH_DB_PATH = os.getenv(
# #     "HASH_DB_PATH",
# #     os.path.join(
# #         os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
# #         "data",
# #         "hash_db.json",
# #     ),
# # )
# # os.makedirs(os.path.dirname(HASH_DB_PATH), exist_ok=True)


# # # ─────────────────────────────────────────────────────────────────
# # # LOAD / SAVE hash DB
# # # ─────────────────────────────────────────────────────────────────
# # def load_hash_db():
# #     if not os.path.exists(HASH_DB_PATH):
# #         return set()
# #     try:
# #         with open(HASH_DB_PATH, "r") as f:
# #             content = f.read().strip()
# #             if not content:
# #                 return set()
# #             return set(json.loads(content))
# #     except json.JSONDecodeError:
# #         print("WARNING: hash_db.json is corrupted. Resetting.")
# #         return set()


# # def save_hash_db(hash_set):
# #     tmp_path = HASH_DB_PATH + ".tmp"
# #     with open(tmp_path, "w") as f:
# #         json.dump(list(hash_set), f, indent=2)
# #     os.replace(tmp_path, HASH_DB_PATH)


# # # ─────────────────────────────────────────────────────────────────
# # # IMAGE METADATA (EXIF) — embed hash + date
# # # ─────────────────────────────────────────────────────────────────
# # _USER_COMMENT_PREFIX = b"ASCII\x00\x00\x00"


# # def _parse_user_comment(raw):
# #     if not raw:
# #         return None
# #     payload = raw[8:] if len(raw) > 8 else raw
# #     try:
# #         text = payload.decode("utf-8", errors="ignore").strip("\x00").strip()
# #         if not text:
# #             return None
# #         return json.loads(text)
# #     except (json.JSONDecodeError, UnicodeDecodeError):
# #         return None


# # def read_image_metadata(raw_bytes: bytes):
# #     """
# #     Read embedded {phash, date} from image EXIF UserComment.
# #     Returns (phash, date_str) or (None, None) if absent.
# #     """
# #     try:
# #         exif_dict = piexif.load(raw_bytes)
# #     except Exception:
# #         return None, None

# #     user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
# #     parsed = _parse_user_comment(user_comment)
# #     if not parsed:
# #         return None, None
# #     return parsed.get("phash"), parsed.get("date")


# # def write_image_metadata(raw_bytes: bytes, phash: str, date_str: str = None):
# #     """
# #     Embed {phash, date} into image EXIF UserComment.
# #     If both are ALREADY present (and phash matches), returns the original bytes.
# #     Otherwise returns NEW bytes with metadata written.
# #     Only works for JPEG. Non-JPEG returns unchanged bytes.
# #     """
# #     if date_str is None:
# #         date_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# #     try:
# #         img = Image.open(io.BytesIO(raw_bytes))
# #         fmt = (img.format or "").upper()
# #     except Exception:
# #         return raw_bytes

# #     if fmt != "JPEG":
# #         return raw_bytes

# #     existing_hash, existing_date = read_image_metadata(raw_bytes)
# #     if existing_hash == phash and existing_date:
# #         return raw_bytes

# #     try:
# #         exif_dict = piexif.load(raw_bytes)
# #     except Exception:
# #         exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

# #     payload = json.dumps({"phash": phash, "date": date_str}).encode("utf-8")
# #     exif_dict.setdefault("Exif", {})[piexif.ExifIFD.UserComment] = (
# #         _USER_COMMENT_PREFIX + payload
# #     )

# #     try:
# #         exif_bytes = piexif.dump(exif_dict)
# #     except Exception:
# #         return raw_bytes

# #     out = io.BytesIO()
# #     img.save(out, format="JPEG", exif=exif_bytes, quality=95)
# #     return out.getvalue()


# # # ─────────────────────────────────────────────────────────────────
# # # S3 SUPPORT — parse URLs and upload back
# # # ─────────────────────────────────────────────────────────────────
# # def parse_s3_url(url: str):
# #     """
# #     Parse an S3 URL into (bucket, key, region) or None if not an S3 URL.

# #     Supports:
# #       https://<bucket>.s3.amazonaws.com/<key>
# #       https://<bucket>.s3.<region>.amazonaws.com/<key>
# #       https://s3.amazonaws.com/<bucket>/<key>
# #       https://s3.<region>.amazonaws.com/<bucket>/<key>
# #       s3://<bucket>/<key>
# #     """
# #     if not url:
# #         return None

# #     if url.startswith("s3://"):
# #         rest = url[5:]
# #         if "/" not in rest:
# #             return None
# #         bucket, key = rest.split("/", 1)
# #         return bucket, key, None

# #     parsed = urlparse(url)
# #     host = parsed.netloc
# #     path = parsed.path.lstrip("/")
# #     if not host or not path:
# #         return None

# #     # Virtual-hosted style: <bucket>.s3[.<region>].amazonaws.com
# #     m = re.match(r"^([^.]+)\.s3(?:[.-]([a-z0-9-]+))?\.amazonaws\.com$", host)
# #     if m:
# #         bucket = m.group(1)
# #         region = m.group(2)
# #         return bucket, path, region

# #     # Path style: s3[.<region>].amazonaws.com/<bucket>/<key>
# #     m = re.match(r"^s3(?:[.-]([a-z0-9-]+))?\.amazonaws\.com$", host)
# #     if m:
# #         region = m.group(1)
# #         if "/" not in path:
# #             return None
# #         bucket, key = path.split("/", 1)
# #         return bucket, key, region

# #     return None


# # def upload_bytes_to_s3(bucket: str, key: str, data: bytes,
# #                        region: str = None, content_type: str = "image/jpeg"):
# #     """
# #     Upload bytes to S3, overwriting the existing key.
# #     Requires boto3 and AWS creds (env vars / IAM role / ~/.aws/credentials).
# #     Returns True on success, False on failure.
# #     """
# #     try:
# #         import boto3
# #         from botocore.exceptions import ClientError, NoCredentialsError
# #     except ImportError:
# #         print("WARNING: boto3 not installed — cannot upload to S3. Run: pip install boto3")
# #         return False

# #     try:
# #         s3_client = boto3.client("s3", region_name=region) if region else boto3.client("s3")
# #         s3_client.put_object(
# #             Bucket=bucket,
# #             Key=key,
# #             Body=data,
# #             ContentType=content_type,
# #         )
# #         return True
# #     except (ClientError, NoCredentialsError) as e:
# #         print(f"ERROR: S3 upload failed for s3://{bucket}/{key} -> {e}")
# #         return False
# #     except Exception as e:
# #         print(f"ERROR: Unexpected S3 upload failure -> {e}")
# #         return False

# import os
# import json
# import io
# import re
# from datetime import datetime, timezone
# from urllib.parse import urlparse
# from PIL import Image, PngImagePlugin
# import imagehash
# import piexif

# # ─────────────────────────────────────────────────────────────────
# # CONFIG — persistent path
# # ─────────────────────────────────────────────────────────────────
# HASH_DB_PATH = os.getenv(
#     "HASH_DB_PATH",
#     os.path.join(
#         os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#         "data",
#         "hash_db.json",
#     ),
# )
# os.makedirs(os.path.dirname(HASH_DB_PATH), exist_ok=True)


# # ─────────────────────────────────────────────────────────────────
# # CLOUDFRONT → S3 BUCKET MAPPING (only needed if you also use CF URLs)
# # ─────────────────────────────────────────────────────────────────
# def _load_cloudfront_map():
#     raw = os.getenv("CLOUDFRONT_BUCKET_MAP", "")
#     if not raw:
#         return {}
#     try:
#         return json.loads(raw)
#     except json.JSONDecodeError:
#         print("WARNING: CLOUDFRONT_BUCKET_MAP is not valid JSON. Ignoring.")
#         return {}


# CLOUDFRONT_BUCKET_MAP = _load_cloudfront_map()


# # ─────────────────────────────────────────────────────────────────
# # LOAD / SAVE hash DB
# # ─────────────────────────────────────────────────────────────────
# def load_hash_db():
#     if not os.path.exists(HASH_DB_PATH):
#         return set()
#     try:
#         with open(HASH_DB_PATH, "r") as f:
#             content = f.read().strip()
#             if not content:
#                 return set()
#             return set(json.loads(content))
#     except json.JSONDecodeError:
#         print("WARNING: hash_db.json is corrupted. Resetting.")
#         return set()


# def save_hash_db(hash_set):
#     tmp_path = HASH_DB_PATH + ".tmp"
#     with open(tmp_path, "w") as f:
#         json.dump(list(hash_set), f, indent=2)
#     os.replace(tmp_path, HASH_DB_PATH)


# # ─────────────────────────────────────────────────────────────────
# # IMAGE METADATA — supports both JPEG (EXIF) and PNG (tEXt chunks)
# # ─────────────────────────────────────────────────────────────────
# # JPEG: stored in EXIF UserComment as JSON
# # PNG:  stored as two tEXt chunks: "phash" and "date"
# #
# # Both use the SAME interface: read_image_metadata() and write_image_metadata().

# _USER_COMMENT_PREFIX = b"ASCII\x00\x00\x00"  # EXIF UserComment charset prefix

# # PNG metadata key names (these appear as text chunks in the PNG file)
# _PNG_KEY_PHASH = "phash"
# _PNG_KEY_DATE = "date"


# def _parse_user_comment(raw):
#     if not raw:
#         return None
#     payload = raw[8:] if len(raw) > 8 else raw
#     try:
#         text = payload.decode("utf-8", errors="ignore").strip("\x00").strip()
#         if not text:
#             return None
#         return json.loads(text)
#     except (json.JSONDecodeError, UnicodeDecodeError):
#         return None


# def _detect_format(raw_bytes: bytes):
#     try:
#         img = Image.open(io.BytesIO(raw_bytes))
#         return (img.format or "").upper()
#     except Exception:
#         return None


# def _read_metadata_jpeg(raw_bytes: bytes):
#     try:
#         exif_dict = piexif.load(raw_bytes)
#     except Exception:
#         return None, None

#     user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
#     parsed = _parse_user_comment(user_comment)
#     if not parsed:
#         return None, None
#     return parsed.get("phash"), parsed.get("date")


# def _read_metadata_png(raw_bytes: bytes):
#     try:
#         img = Image.open(io.BytesIO(raw_bytes))
#         # Pillow exposes tEXt chunks via img.info and img.text
#         info = getattr(img, "text", None) or img.info or {}
#         phash = info.get(_PNG_KEY_PHASH)
#         date = info.get(_PNG_KEY_DATE)
#         if phash and date:
#             return phash, date
#         return None, None
#     except Exception:
#         return None, None


# def read_image_metadata(raw_bytes: bytes):
#     """
#     Read embedded {phash, date} from image metadata.
#     Returns (phash, date_str) or (None, None) if absent.
#     Supports JPEG (via EXIF) and PNG (via tEXt chunks).
#     """
#     fmt = _detect_format(raw_bytes)
#     if fmt == "JPEG":
#         return _read_metadata_jpeg(raw_bytes)
#     if fmt == "PNG":
#         return _read_metadata_png(raw_bytes)
#     return None, None


# def _write_metadata_jpeg(raw_bytes: bytes, phash: str, date_str: str):
#     try:
#         exif_dict = piexif.load(raw_bytes)
#     except Exception:
#         exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

#     payload = json.dumps({"phash": phash, "date": date_str}).encode("utf-8")
#     exif_dict.setdefault("Exif", {})[piexif.ExifIFD.UserComment] = (
#         _USER_COMMENT_PREFIX + payload
#     )

#     try:
#         exif_bytes = piexif.dump(exif_dict)
#     except Exception:
#         return raw_bytes

#     img = Image.open(io.BytesIO(raw_bytes))
#     out = io.BytesIO()
#     img.save(out, format="JPEG", exif=exif_bytes, quality=95)
#     return out.getvalue()


# def _write_metadata_png(raw_bytes: bytes, phash: str, date_str: str):
#     try:
#         img = Image.open(io.BytesIO(raw_bytes))

#         # Build PngInfo, preserving any existing text chunks
#         png_info = PngImagePlugin.PngInfo()
#         existing = getattr(img, "text", None) or {}
#         for k, v in existing.items():
#             if k in (_PNG_KEY_PHASH, _PNG_KEY_DATE):
#                 continue  # we'll re-add these
#             try:
#                 png_info.add_text(k, v)
#             except Exception:
#                 pass

#         png_info.add_text(_PNG_KEY_PHASH, phash)
#         png_info.add_text(_PNG_KEY_DATE, date_str)

#         out = io.BytesIO()
#         img.save(out, format="PNG", pnginfo=png_info, optimize=False)
#         return out.getvalue()
#     except Exception as e:
#         print(f"WARNING: PNG metadata write failed: {e}")
#         return raw_bytes


# def write_image_metadata(raw_bytes: bytes, phash: str, date_str: str = None):
#     """
#     Embed {phash, date} into image metadata.
#     Returns original bytes if metadata is already present and matches phash.
#     Otherwise returns NEW bytes.

#     Supports JPEG (EXIF UserComment) and PNG (tEXt chunks).
#     Other formats: returns unchanged bytes.
#     """
#     if date_str is None:
#         date_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

#     fmt = _detect_format(raw_bytes)
#     if fmt not in ("JPEG", "PNG"):
#         return raw_bytes

#     existing_hash, existing_date = read_image_metadata(raw_bytes)
#     if existing_hash == phash and existing_date:
#         return raw_bytes  # already tagged correctly — leave alone

#     if fmt == "JPEG":
#         return _write_metadata_jpeg(raw_bytes, phash, date_str)
#     if fmt == "PNG":
#         return _write_metadata_png(raw_bytes, phash, date_str)
#     return raw_bytes


# def content_type_for_bytes(raw_bytes: bytes) -> str:
#     """Return the appropriate Content-Type header value for these bytes."""
#     fmt = _detect_format(raw_bytes)
#     if fmt == "JPEG":
#         return "image/jpeg"
#     if fmt == "PNG":
#         return "image/png"
#     if fmt == "WEBP":
#         return "image/webp"
#     if fmt == "GIF":
#         return "image/gif"
#     return "application/octet-stream"


# # ─────────────────────────────────────────────────────────────────
# # URL PARSING — S3 + CloudFront
# # ─────────────────────────────────────────────────────────────────
# def parse_s3_url(url: str):
#     """Resolve URL to (bucket, key, region) or None."""
#     if not url:
#         return None

#     if url.startswith("s3://"):
#         rest = url[5:]
#         if "/" not in rest:
#             return None
#         bucket, key = rest.split("/", 1)
#         return bucket, key, None

#     parsed = urlparse(url)
#     host = parsed.netloc.lower()
#     path = parsed.path.lstrip("/")
#     if not host or not path:
#         return None

#     # CloudFront — look up in map
#     if host.endswith(".cloudfront.net"):
#         cfg = CLOUDFRONT_BUCKET_MAP.get(host)
#         if not cfg or not cfg.get("bucket"):
#             print(f"WARNING: CloudFront host {host!r} not in CLOUDFRONT_BUCKET_MAP")
#             return None
#         bucket = cfg["bucket"]
#         region = cfg.get("region")
#         key_prefix = cfg.get("key_prefix", "")
#         key = (key_prefix.strip("/") + "/" + path).strip("/") if key_prefix else path
#         return bucket, key, region

#     # Virtual-hosted S3: <bucket>.s3[.<region>].amazonaws.com
#     m = re.match(r"^([^.]+)\.s3(?:[.-]([a-z0-9-]+))?\.amazonaws\.com$", host)
#     if m:
#         return m.group(1), path, m.group(2)

#     # Path-style S3: s3[.<region>].amazonaws.com/<bucket>/<key>
#     m = re.match(r"^s3(?:[.-]([a-z0-9-]+))?\.amazonaws\.com$", host)
#     if m:
#         region = m.group(1)
#         if "/" not in path:
#             return None
#         bucket, key = path.split("/", 1)
#         return bucket, key, region

#     return None


# def upload_bytes_to_s3(bucket: str, key: str, data: bytes,
#                        region: str = None, content_type: str = "image/jpeg"):
#     """Upload bytes to S3, overwriting the existing key."""
#     try:
#         import boto3
#         from botocore.exceptions import ClientError, NoCredentialsError
#     except ImportError:
#         print("WARNING: boto3 not installed — cannot upload to S3. Run: pip install boto3")
#         return False

#     try:
#         s3_client = boto3.client("s3", region_name=region) if region else boto3.client("s3")
#         s3_client.put_object(
#             Bucket=bucket,
#             Key=key,
#             Body=data,
#             ContentType=content_type,
#         )
#         return True
#     except (ClientError, NoCredentialsError) as e:
#         print(f"ERROR: S3 upload failed for s3://{bucket}/{key} -> {e}")
#         return False
#     except Exception as e:
#         print(f"ERROR: Unexpected S3 upload failure -> {e}")
#         return False

import os
import json
import io
import re
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse
from PIL import Image, PngImagePlugin
import imagehash
import piexif

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CONFIG — persistent path
# ─────────────────────────────────────────────────────────────────
HASH_DB_PATH = os.getenv(
    "HASH_DB_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "hash_db.json",
    ),
)
os.makedirs(os.path.dirname(HASH_DB_PATH), exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# CLOUDFRONT → S3 BUCKET MAPPING (only needed if you also use CF URLs)
# ─────────────────────────────────────────────────────────────────
def _load_cloudfront_map():
    raw = os.getenv("CLOUDFRONT_BUCKET_MAP", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("CLOUDFRONT_BUCKET_MAP is not valid JSON. Ignoring.")
        return {}


CLOUDFRONT_BUCKET_MAP = _load_cloudfront_map()


# ─────────────────────────────────────────────────────────────────
# LOAD / SAVE hash DB
# ─────────────────────────────────────────────────────────────────
def load_hash_db():
    if not os.path.exists(HASH_DB_PATH):
        return set()
    try:
        with open(HASH_DB_PATH, "r") as f:
            content = f.read().strip()
            if not content:
                return set()
            return set(json.loads(content))
    except json.JSONDecodeError:
        log.warning("hash_db.json is corrupted. Resetting.")
        return set()


def save_hash_db(hash_set):
    tmp_path = HASH_DB_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(list(hash_set), f, indent=2)
    os.replace(tmp_path, HASH_DB_PATH)


# ─────────────────────────────────────────────────────────────────
# IMAGE METADATA — supports both JPEG (EXIF) and PNG (tEXt chunks)
# ─────────────────────────────────────────────────────────────────
# JPEG: stored in EXIF UserComment as JSON
# PNG:  stored as two tEXt chunks: "phash" and "date"
#
# Both use the SAME interface: read_image_metadata() and write_image_metadata().

_USER_COMMENT_PREFIX = b"ASCII\x00\x00\x00"  # EXIF UserComment charset prefix

# PNG metadata key names (these appear as text chunks in the PNG file)
_PNG_KEY_PHASH = "phash"
_PNG_KEY_DATE = "date"


def _parse_user_comment(raw):
    if not raw:
        return None
    payload = raw[8:] if len(raw) > 8 else raw
    try:
        text = payload.decode("utf-8", errors="ignore").strip("\x00").strip()
        if not text:
            return None
        return json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _detect_format(raw_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        return (img.format or "").upper()
    except Exception:
        return None


def _read_metadata_jpeg(raw_bytes: bytes):
    try:
        exif_dict = piexif.load(raw_bytes)
    except Exception:
        return None, None

    user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
    parsed = _parse_user_comment(user_comment)
    if not parsed:
        return None, None
    return parsed.get("phash"), parsed.get("date")


def _read_full_metadata_jpeg(raw_bytes: bytes) -> dict:
    try:
        exif_dict = piexif.load(raw_bytes)
    except Exception:
        return {}
    user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
    return _parse_user_comment(user_comment) or {}


def _read_metadata_png(raw_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        # Pillow exposes tEXt chunks via img.info and img.text
        info = getattr(img, "text", None) or img.info or {}
        phash = info.get(_PNG_KEY_PHASH)
        date = info.get(_PNG_KEY_DATE)
        if phash and date:
            return phash, date
        return None, None
    except Exception:
        return None, None


def _read_full_metadata_png(raw_bytes: bytes) -> dict:
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        return dict(getattr(img, "text", None) or img.info or {})
    except Exception:
        return {}


def read_image_metadata(raw_bytes: bytes):
    """
    Read embedded {phash, date} from image metadata.
    Returns (phash, date_str) or (None, None) if absent.
    Supports JPEG (via EXIF) and PNG (via tEXt chunks).
    """
    fmt = _detect_format(raw_bytes)
    if fmt == "JPEG":
        return _read_metadata_jpeg(raw_bytes)
    if fmt == "PNG":
        return _read_metadata_png(raw_bytes)
    return None, None


def _write_metadata_jpeg(raw_bytes: bytes, phash: str, date_str: str, extra: dict = None):
    try:
        exif_dict = piexif.load(raw_bytes)
    except Exception:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    meta = {"phash": phash, "date": date_str}
    if extra:
        meta.update(extra)
    payload = json.dumps(meta).encode("utf-8")
    exif_dict.setdefault("Exif", {})[piexif.ExifIFD.UserComment] = (
        _USER_COMMENT_PREFIX + payload
    )

    try:
        exif_bytes = piexif.dump(exif_dict)
    except Exception:
        return raw_bytes

    img = Image.open(io.BytesIO(raw_bytes))
    out = io.BytesIO()
    img.save(out, format="JPEG", exif=exif_bytes, quality=95)
    return out.getvalue()


def _write_metadata_png(raw_bytes: bytes, phash: str, date_str: str, extra: dict = None):
    try:
        img = Image.open(io.BytesIO(raw_bytes))

        # Build PngInfo, preserving any existing text chunks except ones we're writing
        png_info = PngImagePlugin.PngInfo()
        reserved = {_PNG_KEY_PHASH, _PNG_KEY_DATE}
        if extra:
            reserved.update(extra.keys())
        existing = getattr(img, "text", None) or {}
        for k, v in existing.items():
            if k in reserved:
                continue
            try:
                png_info.add_text(k, v)
            except Exception:
                pass

        png_info.add_text(_PNG_KEY_PHASH, phash)
        png_info.add_text(_PNG_KEY_DATE, date_str)
        if extra:
            for k, v in extra.items():
                png_info.add_text(k, str(v))

        out = io.BytesIO()
        img.save(out, format="PNG", pnginfo=png_info, optimize=False)
        return out.getvalue()
    except Exception as e:
        log.warning("PNG metadata write failed: %s", e)
        return raw_bytes


def write_image_metadata(raw_bytes: bytes, phash: str, date_str: str = None, extra: dict = None):
    """
    Embed {phash, date, ...extra} into image metadata.
    Returns original bytes if all fields are already present and match.
    Otherwise returns NEW bytes.

    Supports JPEG (EXIF UserComment) and PNG (tEXt chunks).
    Other formats: returns unchanged bytes.
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    fmt = _detect_format(raw_bytes)
    if fmt not in ("JPEG", "PNG"):
        return raw_bytes

    existing = _read_full_metadata_jpeg(raw_bytes) if fmt == "JPEG" else _read_full_metadata_png(raw_bytes)
    if existing.get("phash") == phash and existing.get("date"):
        if not extra or all(existing.get(k) == str(v) for k, v in extra.items()):
            return raw_bytes  # already tagged correctly — leave alone

    if fmt == "JPEG":
        return _write_metadata_jpeg(raw_bytes, phash, date_str, extra)
    if fmt == "PNG":
        return _write_metadata_png(raw_bytes, phash, date_str, extra)
    return raw_bytes


def content_type_for_bytes(raw_bytes: bytes) -> str:
    """Return the appropriate Content-Type header value for these bytes."""
    fmt = _detect_format(raw_bytes)
    if fmt == "JPEG":
        return "image/jpeg"
    if fmt == "PNG":
        return "image/png"
    if fmt == "WEBP":
        return "image/webp"
    if fmt == "GIF":
        return "image/gif"
    return "application/octet-stream"


# ─────────────────────────────────────────────────────────────────
# URL PARSING — S3 + CloudFront
# ─────────────────────────────────────────────────────────────────
def parse_s3_url(url: str):
    """Resolve URL to (bucket, key, region) or None."""
    if not url:
        return None

    if url.startswith("s3://"):
        rest = url[5:]
        if "/" not in rest:
            return None
        bucket, key = rest.split("/", 1)
        return bucket, key, None

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lstrip("/")
    if not host or not path:
        return None

    # CloudFront — look up in map
    if host.endswith(".cloudfront.net"):
        cfg = CLOUDFRONT_BUCKET_MAP.get(host)
        if not cfg or not cfg.get("bucket"):
            log.warning("CloudFront host %r not in CLOUDFRONT_BUCKET_MAP", host)
            return None
        bucket = cfg["bucket"]
        region = cfg.get("region")
        key_prefix = cfg.get("key_prefix", "")
        key = (key_prefix.strip("/") + "/" + path).strip("/") if key_prefix else path
        return bucket, key, region

    # Virtual-hosted S3: <bucket>.s3[.<region>].amazonaws.com
    m = re.match(r"^([^.]+)\.s3(?:[.-]([a-z0-9-]+))?\.amazonaws\.com$", host)
    if m:
        return m.group(1), path, m.group(2)

    # Path-style S3: s3[.<region>].amazonaws.com/<bucket>/<key>
    m = re.match(r"^s3(?:[.-]([a-z0-9-]+))?\.amazonaws\.com$", host)
    if m:
        region = m.group(1)
        if "/" not in path:
            return None
        bucket, key = path.split("/", 1)
        return bucket, key, region

    return None


def upload_bytes_to_s3(bucket: str, key: str, data: bytes,
                       region: str = None, content_type: str = "image/jpeg"):
    """
    Upload bytes to S3, overwriting the existing key.
    Returns dict: {"success": bool, "error": str or None, "error_code": str or None}
    """
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        msg = "boto3 not installed — run: pip install boto3"
        log.warning(msg)
        return {"success": False, "error": msg, "error_code": "BOTO3_NOT_INSTALLED"}

    try:
        s3_client = boto3.client("s3", region_name=region) if region else boto3.client("s3")
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return {"success": True, "error": None, "error_code": None}

    except NoCredentialsError as e:
        msg = (
            "AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "environment variables, or attach an IAM role to the server."
        )
        log.error(msg)
        return {"success": False, "error": msg, "error_code": "NO_CREDENTIALS"}

    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "Unknown")
        message = e.response.get("Error", {}).get("Message", str(e))
        full = f"S3 ClientError [{code}] for s3://{bucket}/{key}: {message}"
        log.error(full)
        return {"success": False, "error": full, "error_code": code}

    except Exception as e:
        msg = f"Unexpected S3 upload failure: {type(e).__name__}: {e}"
        log.error(msg)
        return {"success": False, "error": msg, "error_code": "UNEXPECTED"}