#!/bin/bash

. ./docker_include.sh

set -e

#docker login
docker tag $docker_id/$img_name $docker_id/$img_name:$tag
docker image tag $docker_id/$img_name:$tag $docker_id/$img_name:latest
docker push $docker_id/$img_name:$tag

