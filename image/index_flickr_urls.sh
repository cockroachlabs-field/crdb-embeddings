#!/bin/bash

bc --version > /dev/null 2>&1 || {
  echo "Please install 'bc' and then retry $0"
  exit 1
}

if [[ "$#" -ne 1 ]]
then
  echo "Usage: $0 offset_value (e.g. 300)"
  exit 1
fi

. ./image_env.sh

offset=$1
limit=10000

i=0
for url in $( zcat ./flickr_urls.txt.gz | tail -n +$(( offset + 1 )) | head -$limit )
do
  time ./index_image.py $url
  i=$(( i + 1 ))
  if [[ $(( i % 100 )) -eq 0 ]]
  then
    t_sleep=10
  else
    n=$((1 + $RANDOM % 100))
    t_sleep=$( echo "scale=3; $n/100" | bc )
  fi
  sleep $t_sleep
done

echo
echo "NEXT OFFSET: $(( offset + limit ))"
echo

