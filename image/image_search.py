#!/usr/bin/env python3

import re, sys, os, time, random, io
import logging
import psycopg2
from psycopg2.errors import SerializationFailure
import sqlalchemy
from sqlalchemy import create_engine, text, event, insert, Table, MetaData
import warnings
from sklearn.cluster import KMeans
import joblib
from flask import Flask, render_template, redirect, url_for, abort, request, Response, send_file
import json
import base64
import uuid
import os.path
import pickle
import requests
from pgvector.psycopg2 import register_vector
import resource, platform

# Image search
from PIL import Image
from fastembed import ImageEmbedding

THUMBNAIL_DIM = 192, 192

BLOCK_SIZE = 64 * (1 << 10) # Used when striping the model across > 1 row in blob_store
CHARSET = "utf-8"
kmeans_model = { "read": None, "write": None }

KMEANS_DIM = 512 # FastEmbed Image

print() # Clear previous messages

# Set a memory limit (if running on Linux)
if platform.system() == "Linux":
  mem_limit_mb = int(os.environ.get("MEMORY_LIMIT_MB", "4096"))
  print("mem_limit_mb: {} (set via 'export MEMORY_LIMIT_MB=4096')".format(mem_limit_mb))
  rsrc = resource.RLIMIT_DATA
  mem_limit_bytes = mem_limit_mb * (1 << 20)
  resource.setrlimit(rsrc, (mem_limit_bytes, mem_limit_bytes))
else:
  print("Not on Linux; not setting a memory limit")

kmeans_max_iter = int(os.environ.get("KMEANS_MAX_ITER", "100"))
print("kmeans_max_iter: {} (set via 'export KMEANS_MAX_ITER=25')".format(kmeans_max_iter))

kmeans_verbose = int(os.environ.get("KMEANS_VERBOSE", "0"))
print("kmeans_verbose: {} (set via 'export KMEANS_VERBOSE=1')".format(kmeans_verbose))

skip_kmeans = os.environ.get("SKIP_KMEANS", "False").lower() == "true"
print("skip_kmeans: {} (set via 'export SKIP_KMEANS=False')".format(skip_kmeans))

batch_size = int(os.environ.get("BATCH_SIZE", "512"))
print("batch_size: {} (set via 'export BATCH_SIZE=512')".format(batch_size))

n_clusters = int(os.environ.get("N_CLUSTERS", "500"))
print("n_clusters : {} (set via 'export N_CLUSTERS=50')".format(n_clusters))

train_fraction = float(os.environ.get("TRAIN_FRACTION", "0.5"))
print("train_fraction: {} (set via 'export TRAIN_FRACTION=0.10')".format(train_fraction))

model_file = os.environ.get("MODEL_FILE", "kmeans_images_model.pkl")
print("model_file: {} (set via 'export MODEL_FILE=./model.pkl')".format(model_file))

model_url = os.environ.get("MODEL_FILE_URL", "https://storage.googleapis.com/crl-goddard-text/model.pkl")
print("model_url: {} (set via 'export MODEL_FILE_URL=https://somewhere.com/path/model.pkl')".format(model_url))

n_threads = int(os.environ.get("N_THREADS", "1"))
print("n_threads: {} (set via 'export N_THREADS=10')".format(n_threads))

max_retries = int(os.environ.get("MAX_RETRIES", "3"))
print("max_retries: {} (set via 'export MAX_RETRIES=3')".format(max_retries))

secret = os.environ.get("SECRET", uuid.uuid4().hex)
print("shared secret: {}".format(secret))

blob_store_keep_n_rows = os.environ.get("BLOB_STORE_KEEP_N_ROWS", "3")
print("blob_store_keep_n_rows: {}".format(blob_store_keep_n_rows))

max_hits = os.environ.get("MAX_RESULTS", "4")
print("max_hits: {}".format(max_hits))

log_level = os.environ.get("LOG_LEVEL", "WARN").upper()
logging.basicConfig(
  level=log_level
  , format="[%(asctime)s %(threadName)s] %(message)s"
  , datefmt="%m/%d/%Y %I:%M:%S %p"
)
print("Log level: {} (export LOG_LEVEL=[DEBUG|INFO|WARN|ERROR] to change this)".format(log_level))

db_url = os.getenv("DB_URL")
if db_url is None:
  print("DB_URL must be set")
  sys.exit(1)

db_url = re.sub(r"^postgres(ql)?", "cockroachdb", db_url)
engine = create_engine(db_url, pool_size=20, pool_pre_ping=True, connect_args = { "application_name": "CRDB Image Embed" })

@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
  register_vector(dbapi_connection)
  cur = dbapi_connection.cursor()
  cur.execute("SET plan_cache_mode = auto;")
  cur.close()

t0 = time.time()
fe_model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision") # 512 features
et = time.time() - t0
logging.info("ImageEmbedding model ready: {} s".format(et))

# Used to download a model if none exists on FS or in DB
def download_file(url, local_fname):
  with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(local_fname, "wb") as f:
      for chunk in r.iter_content(chunk_size=8192): 
        f.write(chunk)

ddl_schema = """
CREATE SCHEMA IF NOT EXISTS image;
"""

ddl_t1 = """
CREATE TABLE image.image_embed
(
  uri STRING NOT NULL
  , embedding VECTOR ({})
  , PRIMARY KEY (uri)
);
""".format(KMEANS_DIM)

ddl_t2 = """
CREATE TABLE image.cluster_assign
(
  uri STRING NOT NULL
  , cluster_id INT8 NOT NULL
  , PRIMARY KEY (uri)
  , INDEX (cluster_id ASC)
);
"""

# Schema is provided in the string passed into the '{}'
ddl_t3 = """
CREATE TABLE {}
(
  uri STRING NOT NULL
  , cluster_id INT8 NOT NULL
  , PRIMARY KEY (uri)
  , INDEX (cluster_id ASC)
);
"""

ddl_t4 = """
CREATE TABLE image.blob_store
(
  path STRING NOT NULL
  , ts TIMESTAMP NOT NULL DEFAULT now()
  , n_row INT NOT NULL
  , blob BYTEA NOT NULL
  , PRIMARY KEY (path, ts, n_row)
);
"""

ddl_t5 = """
CREATE TABLE image.thumbnail
(
  uri STRING NOT NULL
  , blob BYTEA NOT NULL
  , PRIMARY KEY (uri)
);
"""

ddl_view = """
CREATE OR REPLACE VIEW image.ie_ca_view
AS
(
  SELECT ie.uri, ie.embedding, c.cluster_id
  FROM image.image_embed ie, image.cluster_assign c
  WHERE ie.uri = c.uri
);
"""

sql_check_exists = """
SELECT COUNT(*) n FROM [SHOW TABLES] WHERE table_name = 'image_embed';
"""

image_embed_table = None # Will be set after running setup_db()
cluster_assign_table = None

def run_ddl(ddl):
  with engine.connect() as conn:
    conn.execute(text(ddl))
    conn.commit()

def prune_blob_store():
  logging.info("Pruning blob_store table ...")
  sql = """
  DELETE FROM image.blob_store
  WHERE (path, ts) IN
  (
    SELECT path, ts
    FROM image.blob_store
    GROUP BY 1, 2
    ORDER BY 2 DESC
    OFFSET {}
  );
  """
  with engine.connect() as conn:
    conn.execute(text(sql.format(blob_store_keep_n_rows)))
    conn.commit()
  logging.info("OK")

def setup_db():
  logging.info("Checking whether image_embed table exists")
  n_rows = 0
  with engine.connect() as conn:
    rs = conn.execute(text(sql_check_exists))
    for row in rs:
      n_rows = row.n
  table_exists = (n_rows == 1)
  if not table_exists:
    logging.info("Creating image tables and view ...")
    run_ddl(ddl_schema)
    run_ddl(ddl_t1)
    run_ddl(ddl_t2)
    run_ddl(ddl_t4)
    run_ddl(ddl_t5)
    run_ddl(ddl_view)
    logging.info("OK")
  else:
    logging.info("image_embed table already exists")

# Retry wrapper for functions interacting with the DB
def retry(f, args):
  for n_retry in range(0, max_retries):
    if n_retry > 0:
      logging.warning("Retry number {}".format(n_retry))
    try:
      return f(*args)
    except SerializationFailure as e:
      logging.warning("Error: %s", e)
      logging.warning("EXECUTE SERIALIZATION_FAILURE BRANCH")
      sleep_s = (2**n_retry) * 0.1 * (random.random() + 0.5)
      logging.warning("Sleeping %s s", sleep_s)
      time.sleep(sleep_s)
    except (sqlalchemy.exc.OperationalError, psycopg2.OperationalError) as e:
      # Get a new connection and try again
      logging.warning("Error: %s", e)
      logging.warning("EXECUTE CONNECTION FAILURE BRANCH")
      sleep_s = 0.12 + random.random() * 0.25
      logging.warning("Sleeping %s s", sleep_s)
      time.sleep(sleep_s)
      if isinstance(args[0], sqlalchemy.engine.base.Connection):
        logging.warning("Getting a new Connection instance ...")
        args[0] = engine.connect()
    except psycopg2.Error as e:
      logging.warning("Error: %s", e)
      logging.warning("EXECUTE DEFAULT BRANCH")
      raise e
  raise ValueError(f"Transaction did not succeed after {max_retries} retries")

# Return cluster ID value for embedding using "read" or "write" model
def get_cluster_id(rw, embed):
  return int(kmeans_model[rw].predict([embed])[0])

def get_image_from_url(url):
  rv = None
  with Image.open(requests.get(url, stream=True).raw) as img:
    rv = img.copy()
  return rv

# Given an Image instance, return a thumbnail version of that Image
def gen_thumbnail(img):
  thumb = img.copy()
  thumb.thumbnail(THUMBNAIL_DIM)
  return thumb

def get_image_features(img):
  embed = list(fe_model.embed([img]))
  return embed[0]

def index_image(conn, uri):
  t0 = time.time()
  embed = None
  thumb = None
  try:
    img = get_image_from_url(uri)
    embed = get_image_features(img)
    thumb = gen_thumbnail(img)
  except Exception as e:
    logging.warning(e)
    return Response(str(e), status=500, mimetype="text/plain")
  et = time.time() - t0
  logging.info("Time to generate embeddings(): {} ms".format(et * 1000))
  ie_row_map = {
    "uri": uri
    , "embedding": embed
  }
  img_io = io.BytesIO()
  thumb.save(img_io, "JPEG")
  img_io.seek(0)
  th_row_map = {
    "uri": uri
    , "blob": img_io.read()
  }
  if not skip_kmeans:
    cluster_id = get_cluster_id("write", embed)
    ca_row_map = {
        "uri": uri
        , "cluster_id": cluster_id
    }
  t0 = time.time()
  try:
    conn.execute(insert(image_embed_table), [ie_row_map])
    conn.execute(insert(thumb_table), [th_row_map])
  except sqlalchemy.exc.IntegrityError as e:
    return Response(e.orig.diag.message_detail, status=400, mimetype="text/plain")
  if not skip_kmeans:
    conn.execute(insert(cluster_assign_table), [ca_row_map])
  conn.commit()
  et = time.time() - t0
  logging.info("DB INSERT time: {} ms".format(et * 1000))
  return Response("OK", status=200, mimetype="text/plain")

# Decode a base64 encoded value to a UTF-8 string
def decode(b64):
  b = base64.b64decode(b64)
  return b.decode(CHARSET).strip()

app = Flask(__name__)

def gen_sql():
  rv = """
WITH q_embed AS
(
  SELECT uri, embedding
  FROM image.ie_ca_view
  WHERE cluster_id = :cluster_id
)
SELECT uri, 1 - (embedding <=> (:q_embed)::VECTOR) sim
FROM q_embed
ORDER BY sim DESC
LIMIT :limit
"""
  return rv

def verify_secret(s):
  err = None
  if s != secret:
    err = "Provided secret '{}' != expected value '{}'".format(s, secret)
    logging.warning(err)
  return err

def refresh_cluster_assignments(s):
  err = verify_secret(s)
  if err is not None:
    return Response(err, status=400, mimetype="text/plain")
  # Temporary table to insert mappings into
  temp_table_name = "cluster_assign_temp_{}".format(uuid.uuid4().hex)
  logging.info("Inserting cluster assignments into {}".format(temp_table_name))
  run_ddl(ddl_t3.format("image." + temp_table_name))
  cluster_assign_table_new = Table(temp_table_name, db_meta, autoload_with=engine, extend_existing=True)
  select_sql = """
  SELECT uri, embedding
  FROM image.image_embed
  WHERE uri > :last_uri
  ORDER BY 1
  LIMIT :limit;
  """
  t0 = time.time()
  with engine.connect() as conn:
    ins_list = []
    uri = ''
    chunk_num = -1
    while True:
      stmt = text(select_sql).bindparams(last_uri=uri, limit=batch_size)
      rs = conn.execute(stmt)
      if rs is None or rs.rowcount == 0:
        break
      for row in rs:
        (uri, embed) = row
        cluster_id = get_cluster_id("write", embed)
        row_map = {
          "uri": uri
          , "cluster_id": cluster_id
        }
        ins_list.append(row_map)
      logging.info("Inserting batch of {} rows".format(batch_size))
      conn.execute(insert(cluster_assign_table_new), ins_list)
      conn.commit()
      ins_list = []
  et = time.time() - t0
  logging.info("Cluster assign time: {} ms".format(et * 1000))
  # Swap the tables
  t0 = time.time()
  logging.info("Swapping the tables for cluster_assign ...")
  with engine.connect() as conn:
    conn.execute(text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;"))
    conn.execute(text("DROP VIEW image.ie_ca_view;"))
    conn.execute(text("DROP TABLE image.cluster_assign;"))
    conn.execute(text("ALTER TABLE image.{} RENAME TO image.cluster_assign;".format(temp_table_name)))
    conn.execute(text(ddl_view))
    conn.commit()
  et = time.time() - t0
  logging.info("Table swap time: {} ms".format(et * 1000))
  kmeans_model["read"] = kmeans_model["write"] # Once cluster assignments are updated
  return Response("OK", status=200, mimetype="text/plain")

def fetch_thumb(uri):
  sql = """
  SELECT blob
  FROM image.thumbnail
  WHERE uri = :uri
  """
  rv = None
  with engine.connect() as conn:
    rs = conn.execute(text(sql).bindparams(uri=uri))
    for row in rs:
      rv = row.blob
  return rv

@app.route("/thumb/<uri_base_64>")
def thumb(uri_base_64):
  uri = decode(uri_base_64)
  image_data = fetch_thumb(uri)
  image_stream = io.BytesIO(image_data)
  return send_file(image_stream, mimetype="image/jpeg")

@app.route("/cluster_assign/<s>")
def cluster_assign(s):
  return retry(refresh_cluster_assignments, (s,))

# Store the model to the DB
def store_model_in_db(mdl):
  rows = []
  orig_io = io.BytesIO(pickle.dumps(mdl))
  chunk = orig_io.read(BLOCK_SIZE)
  while chunk:
    row_map = {
      "path": model_file
      , "n_row": len(rows)
      , "blob": chunk
    }
    rows.append(row_map)
    chunk = orig_io.read(BLOCK_SIZE)
  with engine.begin() as conn:
    conn.execute(insert(blob_table), rows)

@app.route("/sample/<int:n_rows>")
def sample_data(n_rows):
  t0 = time.time()
  logging.info("Getting {} sample rows ...".format(n_rows))
  # Use an efficient approach to get a set of keys to select from
  sql = """
  SHOW STATISTICS USING JSON FOR TABLE image.cluster_assign;
  """
  stmt = text(sql)
  js = None
  with engine.connect() as conn:
    conn.execute(text("SET TRANSACTION AS OF SYSTEM TIME '-10s';"))
    rs = conn.execute(stmt)
    if rs is not None:
      for row in rs:
        js = row[0]
  keys = []
  keys.append('') # Default value if no stats exist
  for v1 in js:
    if v1["columns"] == ["uri"]:
      for hb in v1["histo_buckets"]:
        keys.append(hb["upper_bound"])
  # Fetch the sample rows using a randomly chosen key from above list
  key = random.choice(keys)
  sql = """
  SELECT uri
  FROM image.image_embed
  WHERE uri > :key
  LIMIT :limit;
  """
  stmt = text(sql).bindparams(key=key, limit=n_rows)
  sample = []
  with engine.connect() as conn:
    conn.execute(text("SET TRANSACTION AS OF SYSTEM TIME '-10s';"))
    rs = conn.execute(stmt)
    if rs is not None:
      for row in rs:
        sample.append(row[0])
  et = time.time() - t0
  logging.info("SQL query time: {} ms".format(et * 1000))
  return Response(json.dumps(sample), status=200, mimetype="application/json")

@app.route("/build_model/<s>")
def build_model(s):
  global kmeans_model
  err = verify_secret(s)
  if err is not None:
    return Response(err, status=400, mimetype="text/plain")
  logging.info("Getting data sample for model build ...")
  # Grab a sample of vectors
  sql = """
  SELECT embedding
  FROM image.image_embed
  WHERE random() < :fraction
  """
  t0 = time.time()
  stmt = text(sql).bindparams(fraction=train_fraction)
  sampled_vecs = []
  with engine.connect() as conn:
    conn.execute(text("SET TRANSACTION AS OF SYSTEM TIME '-10s';"))
    with conn.execution_options(stream_results=True, max_row_buffer=batch_size).execute(stmt) as rs:
      for row in rs:
        sampled_vecs.append(row[0])
  et = time.time() - t0
  logging.info("SQL query time: {} ms".format(et * 1000))
  kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=137,
    init="k-means++",
    n_init=10,
    max_iter=kmeans_max_iter,
    verbose=kmeans_verbose,
    algorithm="elkan"
  )
  logging.info("Starting model build ...")
  t0 = time.time()
  model = kmeans.fit(sampled_vecs)
  et = time.time() - t0
  logging.info("Model build time: {} ms".format(et * 1000))
  # Store the model to the filesystem
  joblib.dump(model, model_file)
  store_model_in_db(model)
  # Reload the in-memory copy of the model, so any writes will use updated model
  kmeans_model["write"] = model
  prune_blob_store()
  return Response("OK", status=200, mimetype="text/plain")

# Returns: list of {"uri": uri, "score": sim }
def search(conn, uri, img, limit):
  logging.info("Query URI: '{}'".format(uri))
  rv = []
  t0 = time.time()
  embed = get_image_features(img)
  et = time.time() - t0
  logging.info("  get_image_features(): {} ms".format(et * 1000))
  t0 = time.time()
  cluster_id = get_cluster_id("read", embed) # This works fine with the ndarray type
  et = time.time() - t0
  logging.info("  cluster ID: {}, time: {} ms".format(cluster_id, et * 1000))
  t0 = time.time()
  stmt = text(gen_sql()).bindparams(q_embed=embed, cluster_id=cluster_id, limit=limit)
  rs = conn.execute(stmt)
  if rs is not None:
    for row in rs:
      (uri, sim) = row
      rv.append({"uri": uri, "score": float(sim)})
  et = time.time() - t0
  logging.info("  SQL query time: {} ms".format(et * 1000))
  return rv

# Pass the image URL, base64 encoded
@app.route("/search/<int:n_items>/<url_b64>")
def query_image_url(n_items, url_b64):
  url = decode(url_b64)
  img = get_image_from_url(url)
  with engine.connect() as conn:
    rv = retry(search, (conn, url, img, n_items))
  return Response(json.dumps(rv), status=200, mimetype="application/json")

@app.route("/index/<url_b64>")
def do_index(url_b64):
  url = decode(url_b64)
  with engine.connect() as conn:
    rv = retry(index_image, (conn, url))
  return rv

@app.route("/health", methods=["GET"])
def health():
  return Response("OK", status=200, mimetype="text/plain")

# Fetch most recent model from the DB
def get_model_from_db():
  logging.info("Fetching model from the DB ...")
  sql = """
  WITH u AS
  (
    SELECT path, ts
    FROM image.blob_store
    ORDER BY ts DESC
    LIMIT 1
  )
  SELECT b.blob blob
  FROM image.blob_store b, u
  WHERE b.path = u.path AND b.ts = u.ts
  ORDER BY b.n_row ASC;
  """
  rv = None
  buf = io.BytesIO()
  with engine.connect() as conn:
    rs = conn.execute(text(sql))
    for row in rs:
      buf.write(row.blob)
  blob = buf.getvalue()
  if len(blob) > 0:
    rv = pickle.loads(blob)
    logging.info("OK")
  else:
    logging.info("No model in the DB")
  return rv

# main()
setup_db()
db_meta = MetaData(schema="image")
with warnings.catch_warnings():
  warnings.simplefilter("ignore", category=sqlalchemy.exc.SAWarning)
  image_embed_table = Table("image_embed", db_meta, autoload_with=engine)
cluster_assign_table = Table("cluster_assign", db_meta, autoload_with=engine)
blob_table = Table("blob_store", db_meta, autoload_with=engine)
thumb_table = Table("thumbnail", db_meta, autoload_with=engine)

# Load the K-means model
if not skip_kmeans:
  model_from_db = get_model_from_db()
  if model_from_db is None:
    if not os.path.isfile(model_file):
      logging.info("Downloading bootstrap K-means file ...")
      logging.info("\tURL: {}".format(model_url))
      logging.info("\tLocal file: {}".format(model_file))
      download_file(model_url, model_file)
      logging.info("OK")
    # Now the file is on the local FS, so load it and store it
    kmeans_model["read"] = joblib.load(model_file)
    kmeans_model["write"] = kmeans_model["read"]
    store_model_in_db(kmeans_model["read"])
  else:
    kmeans_model["write"] = model_from_db
    kmeans_model["read"] = model_from_db
  logging.info("K-means model loaded")
  logging.info("You may need to update K-means cluster assignments by making a GET request to the /cluster_assign/{} endpoint.".format(secret))
else:
  logging.info("Skipping loading K-Means model")

# File uploads (https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask)
app.config["MAX_CONTENT_LENGTH"] = 5 * (1 << 20)
app.config["UPLOAD_PATH"] = "/tmp/uploads"

@app.errorhandler(413)
def too_large(e):
  return "File is too large", 413

@app.route('/')
def index():
 files = os.listdir(app.config["UPLOAD_PATH"])
 return render_template("index.html", hits=files)

def b64_encode(s):
  return base64.b64encode(s.encode("utf-8")).decode("utf-8")

def gen_thumb_path(fn):
  return os.path.join(app.config["UPLOAD_PATH"], '.'.join(fn.split('.')[0:-1]) + ".thumb.jpg")

@app.route("/thumb_fs/<filename>")
def thumb_from_file(filename):
  image_data = None
  with open(gen_thumb_path(filename), mode="rb") as f:
    image_data = f.read()
  image_stream = io.BytesIO(image_data)
  return send_file(image_stream, mimetype="image/jpeg")

@app.route('/', methods=["POST"])
def upload_files():
  rv = []
  uploaded_file = request.files["file"]
  filename = uploaded_file.filename
  if filename == '':
    return render_template("index.html", hits=rv)
  content = uploaded_file.read()
  img = Image.open(io.BytesIO(content))
  logging.info("{}: {} bytes".format(filename, str(len(content))))
  img_uri = "file:///{}".format(filename)
  with engine.connect() as conn:
    rv = retry(search, (conn, img_uri, img, max_hits))
    for hit in rv:
      hit["uri_b64"] = b64_encode(hit["uri"])
      hit["score"] = "{:.3f}".format(float(hit["score"]))
  # TODO: poll these files and index them async?
  uploaded_file.seek(0)
  uploaded_file.save(os.path.join(app.config["UPLOAD_PATH"], filename))
  thumb = gen_thumbnail(img)
  thumb.save(gen_thumb_path(filename), "JPEG")
  return render_template("index.html", hits=rv, search_file=filename)

# __main__
port = int(os.getenv("FLASK_PORT", 18080))
os.makedirs(app.config["UPLOAD_PATH"], mode=0o755, exist_ok=True)
from waitress import serve
serve(app, host="0.0.0.0", port=port, threads=n_threads)

