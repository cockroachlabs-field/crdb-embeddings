# For server
#export FLASK_PORT=1972
#export FLASK_HOST=localhost
export FLASK_PORT=80
export FLASK_HOST=vector.la-cucaracha.net
#export DB_URL="postgres://test_role:123abc@127.0.0.1:26257/defaultdb?sslmode=require&sslrootcert=$HOME/certs/ca.crt"
export DB_URL="postgres://test_role:123abc@127.0.0.1:26257/embed?sslmode=require&sslrootcert=$HOME/certs/ca.crt"

export LOG_LEVEL=INFO
export N_THREADS=10
export CACHE_SIZE=1024
export MIN_SENTENCE_LEN=8
export N_CLUSTERS=100
export TRAIN_FRACTION=0.75
export MODEL_FILE=/tmp/model.pkl
export MODEL_FILE_URL="https://storage.googleapis.com/crl-goddard-text/model.pkl"
export BATCH_SIZE=512
export KMEANS_VERBOSE=1
export KMEANS_MAX_ITER=25
export SECRET="TextWithNoSpecialChars"
export BLOB_STORE_KEEP_N_ROWS=3

# For client
export MAX_RESULTS=5

