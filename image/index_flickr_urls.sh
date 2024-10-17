#!/bin/bash

if [[ "$#" -ne 1 ]]
then
  echo "Usage: $0 offset_value (e.g. 300)"
  exit 1
fi

offset=$1
limit=100

for url in $( gzcat ./flickr_urls.txt.gz | tail -n +$(( offset + 1 )) | head -$limit )
do
  time ./index_image.py $url
done

echo
echo "NEXT OFFSET: $(( offset + limit ))"
echo
