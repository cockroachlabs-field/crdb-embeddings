#!/bin/bash

lb_ip=$( kubectl describe service crdb-lb | perl -ne 'chomp; print "$1\n" if /^LoadBalancer Ingress:\s+((\d+\.){3}\d+)/;' )

cert="/tmp/ca.crt"

# Remove any stale cert file
find $cert -mmin +20 -exec rm -f {} \; >/dev/null 2>&1

# Get the CA cert if necessary
[ -f $cert ] || ./get_ca_cert.sh

export PAGER=cat # Disable paging
psql "postgresql://embeduser:embedpasswd@${lb_ip}:26257/defaultdb?sslmode=require&sslrootcert=$cert"

