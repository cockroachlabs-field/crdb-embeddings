#!/usr/bin/env python

# Store embeddings as STRING[] and use && (indexed) for boolean "is a match"
# and UDF overlap() for scoring.

import torch
import re, sys, os, time
from transformers import BertTokenizer, BertModel
import base36
import logging
import psycopg2

# Max number of dimensions to store in DB and use for queries (out of 768)
#TOP_N = 32
#TOP_N = 10
#TOP_N = 8
#TOP_N = 16
TOP_N = 64

# Discard the first N tokens as they have little differentiating value
"""
defaultdb=> select substring(token from 1 for 6), count(*) from text_embed group by 1 order by 2 desc;
 substring | count
-----------+-------
 8k al     |    30
 8k ez     |    22
 8k 69     |     2
 8k in     |     1
 8k cd     |     1
 8k 2b     |     1
 8k 50     |     1
(7 rows)
"""
N_DISCARD = 2

# Delimiter for the base36 encoded array dimension values
DELIM = ' '
#DELIM = '~'

# Tweak minimum similarity in the DB session.  Lower may be better given the LIMIT clause.
# set pg_trgm.similarity_threshold = 0.25;
# set pg_trgm.similarity_threshold = 0.1;
MIN_SIM = 0.1 # Will get set in SQL session

log_level = os.environ.get("LOG_LEVEL", "WARN").upper()
logging.basicConfig(
  level=log_level
  , format="[%(asctime)s] %(message)s"
  , datefmt="%m/%d/%Y %I:%M:%S %p"
)
print("Log level: {} (export LOG_LEVEL=[DEBUG|INFO|WARN|ERROR] to change this)".format(log_level))

db_url = os.getenv("DB_URL")
if db_url is None:
  print("DB_URL must be set")
  sys.exit(1)

if len(sys.argv) < 2:
  print("Usage: {} file [file2 ...]".format(sys.argv[0]))
  sys.exit(1)

t0 = time.time()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
et = time.time() - t0
logging.info("BertTokenizer: {} s".format(et))

t0 = time.time()
# Set this up once and reuse
model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
model.eval()
et = time.time() - t0
logging.info("BertModel + eval: {} s".format(et))

# The fist call to this takes ~ 500 ms but subsequent calls take ~ 40 ms
def gen_embeddings(s):
  rv = None
  marked_text = "[CLS] " + s + " [SEP]"
  tokenized_text = tokenizer.tokenize(marked_text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_ids = [1] * len(tokenized_text)
  segments_tensors = torch.tensor([segments_ids])
  with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]
  token_vecs = hidden_states[-2][0]
  sentence_embedding = torch.mean(token_vecs, dim=0)
  rv = sentence_embedding.tolist()
  return rv

# From the list of embeddings, the 768 element array, return an array of string
# consisting of the base 36 encoded array dimension (2 chars) for the TOP_N
# elements having the largest magnitude.
def gen_embed_token(embed_list):
  dims = {}
  for i in range(0, len(embed_list)):
    dims[base36.dumps(i).zfill(2)] = embed_list[i]
  return list(dict(sorted(dims.items(), key=lambda item: abs(item[1]), reverse=True)[N_DISCARD:TOP_N + N_DISCARD]).keys())

def get_token_for_string(s):
  rv = None
  t0 = time.time()
  embed = gen_embeddings(s)
  et = time.time() - t0
  logging.info("gen_embeddings: {} s".format(et))
  return gen_embed_token(embed)

"""
DROP TABLE IF EXISTS text_embed;
CREATE TABLE text_embed
(
  uri STRING NOT NULL
  , chunk_num INT NOT NULL
  , token STRING[] NOT NULL
  , chunk STRING NOT NULL
  , PRIMARY KEY (uri, chunk_num)
);
CREATE INDEX ON text_embed USING GIN (token);
"""

ins_sql = "INSERT INTO text_embed (uri, chunk_num, token, chunk) VALUES (%s, %s, %s, %s)"
def index_file(in_file):
  text = ""
  with open(in_file, mode="rt") as f:
    for line in f:
      text += line
  in_file = re.sub(r"\./", '', in_file)
  with conn.cursor() as cur:
    n_chunk = 0
    for s in re.split(r"\. +", text):
      s = s.strip()
      if (len(s) > 0):
        token = get_token_for_string(s)
        cur.execute(ins_sql, (in_file, n_chunk, token, s))
        n_chunk += 1
    conn.commit()

q_sql = """
WITH q_embed AS
(
  SELECT uri, SIMILARITY(%s, token)::NUMERIC(4, 3) sim, token, chunk
  FROM text_embed@text_embed_token_idx
  WHERE %s %% token
  ORDER BY sim DESC
  LIMIT 5
)
SELECT * from q_embed where chunk ~* %s;
"""
# SELECT * from q_embed where chunk ~* '(%s)';

conn = psycopg2.connect(db_url)

# Query mode
if "-q" == sys.argv[1][0:2]:
  terms = sys.argv[2:]
  q = ' '.join(terms)
  tok = get_token_for_string(q)
  print("Query string: '{}'\nToken: '{}'\n".format(q, tok))
  terms_regex = '({})'.format('|'.join(list(set(terms)))) # Remove duplicate terms via the set
  print("terms_regex: {}\n".format(terms_regex))
  t0 = time.time()
  with conn.cursor() as cur:
    cur.execute("SET pg_trgm.similarity_threshold = %s;", (MIN_SIM,)) # This does work
  conn.commit()
  with conn.cursor() as cur:
    cur.execute(q_sql, (tok, tok, terms_regex,))
    rs = cur.fetchall()
    if rs is not None:
      for row in rs:
        (uri, sim, token, chunk) = row
        print("URI: {}\nSCORE: {}\nTOKEN: {}\nCHUNK: {}\n".format(uri, sim, token, chunk))
    else:
      print("Empty result set\n")
  et = time.time() - t0
  print("SQL query time: {} ms\n".format(et * 1000))
else:
  # Indexing mode
  t0 = time.time()
  for in_file in sys.argv[1:]:
    print("Indexing file " + in_file + " now ...")
    index_file(in_file)
  et = time.time() - t0
  logging.info("Total time: {} s".format(et))

# Close connection
conn.close()

