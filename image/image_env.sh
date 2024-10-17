# For server
export FLASK_PORT=1972
export FLASK_HOST=localhost
#export FLASK_PORT=80
#export FLASK_HOST=vector.la-cucaracha.net
export DB_URL="postgres://test_role:123abc@127.0.0.1:26257/defaultdb"

export LOG_LEVEL=INFO
export N_THREADS=10
export N_CLUSTERS=1024
export TRAIN_FRACTION=0.3
export MODEL_FILE=/tmp/image_model.pkl
export MODEL_FILE_URL="https://storage.googleapis.com/crl-goddard-text/image_model_1024_25k.pkl"
export BATCH_SIZE=256
export KMEANS_VERBOSE=2
export KMEANS_MAX_ITER=100
export SECRET="TextWithNoSpecialChars"
export BLOB_STORE_KEEP_N_ROWS=3
export MEMORY_LIMIT_MB=16384
export SKIP_KMEANS=false

# For client
export MAX_RESULTS=5

