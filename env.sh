# For server
export FLASK_PORT=1972
export FLASK_HOST=localhost
export DB_URL="postgres://test_role:123abc@127.0.0.1:26257/defaultdb?sslmode=require&sslrootcert=$HOME/certs/ca.crt"

export LOG_LEVEL=INFO
export N_THREADS=1
export CACHE_SIZE=1024
export MIN_SENTENCE_LEN=8
export N_CLUSTERS=50
export TRAIN_FRACTION=0.25
export MODEL_FILE=/tmp/model.pkl
export BATCH_SIZE=512

# For client
export MAX_RESULTS=6
export TOP_N_Q=8

