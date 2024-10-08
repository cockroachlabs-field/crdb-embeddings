#!/bin/bash

# https://www.cockroachlabs.com/docs/stable/deploy-cockroachdb-with-kubernetes.html

MACHINETYPE="e2-standard-4" # 4	vCPU, 16 GB RAM, $0.134012/hour
N_NODES=2 # This will create N_NODES *per AZ* within REGION
REGION="us-east4"

NAME="${USER}-crdb-embeddings"

dir=$( dirname $0 )
. $dir/include.sh

# Create the GKE K8s cluster
export USE_GKE_GCLOUD_AUTH_PLUGIN=True
echo "See https://www.cockroachlabs.com/docs/v21.1/orchestrate-cockroachdb-with-kubernetes.html#hosted-gke"
run_cmd gcloud container clusters create $NAME --region=$REGION --machine-type=$MACHINETYPE --num-nodes=$N_NODES
if [ "$y_n" = "y" ] || [ "$y_n" = "Y" ]
then
  ACCOUNT=$( gcloud info | perl -ne 'print "$1\n" if /^Account: \[([^@]+@[^\]]+)\]$/' )
  kubectl create clusterrolebinding $USER-cluster-admin-binding --clusterrole=cluster-admin --user=$ACCOUNT
fi

# Create the CockroachDB cluster
echo "See https://www.cockroachlabs.com/docs/stable/deploy-cockroachdb-with-kubernetes.html"
echo "Apply the CustomResourceDefinition (CRD) for the Operator"
run_cmd kubectl apply -f https://raw.githubusercontent.com/cockroachdb/cockroach-operator/v2.10.0/install/crds.yaml

echo "Apply the Operator manifest"
OPERATOR_YAML="./operator.yaml"
curl https://raw.githubusercontent.com/cockroachdb/cockroach-operator/v2.10.0/install/operator.yaml | sed 's/namespace: cockroach-operator-system/namespace: default/g' > $OPERATOR_YAML
run_cmd kubectl apply -f $OPERATOR_YAML

echo "Validate that the Operator is running"
run_cmd kubectl get pods

echo "Initialize the cluster"
run_cmd kubectl apply -f $dir/cockroachdb.yaml

echo "Check that the pods were created"
run_cmd kubectl get pods

echo "WAIT until the output of 'kubectl get pods' shows the three cockroachdb-N nodes in 'Running' state"
echo "(This could take upwards of 5 minutes)"
run_cmd kubectl get pods

echo "Check to see whether the LB for DB Console and SQL is ready yet"
echo "Look for the external IP of the app in the 'LoadBalancer Ingress:' line of output"
run_cmd kubectl describe service crdb-lb
echo "If not, run 'kubectl describe service crdb-lb' in a separate window"

# Deploy a SQL client
#SQL_CLIENT_YAML="https://raw.githubusercontent.com/cockroachdb/cockroach-operator/master/examples/client-secure-operator.yaml"
SQL_CLIENT_YAML="https://raw.githubusercontent.com/cockroachdb/cockroach-operator/v2.10.0/examples/client-secure-operator.yaml"
echo "Adding a secure SQL client pod ..."
kubectl create -f $SQL_CLIENT_YAML
echo "Done"

echo "Verify the 'cockroachdb-client-secure' is in 'Running' state"
kubectl get pods
sleep 5
kubectl get pods

# Add DB user for app
echo "Once all three DB pods show 'Running', use the SQL CLI to add a user for use by the Web app"
echo "Press ENTER to run this SQL"
read
cat $dir/create_user.sql | kubectl exec -i cockroachdb-client-secure -- ./cockroach sql --certs-dir=/cockroach/cockroach-certs --host=cockroachdb-public

# Start the CockroachDB DB Console
echo "Open a browser tab to port 8080 at the IP provided for the DB Console endpoint"
echo "** Use 'embeduser' as login with password 'embedpasswd' **"

# Start the Web app
echo "Press ENTER to start the CockroachDB Embeddings app"
read
kubectl apply -f $dir/crdb-embeddings.yaml

# Get the IP address of the load balancer
run_cmd kubectl describe service crdb-embeddings-lb
echo "Look for the external IP of the app in the 'LoadBalancer Ingress:' line of output"
sleep 30
run_cmd kubectl describe service crdb-embeddings-lb
echo "Once that IP is available, open the URL http://THIS_IP/ to see the app running"
echo

# Kill a node
echo "Kill a CockroachDB pod"
run_cmd kubectl delete pods cockroachdb-0
echo "Reload the app page to verify it continues to run"
echo "Also, note the state in the DB Console"
echo "A new pod should be started to replace the failed pod"
run_cmd kubectl get pods

# Scale out by adding a 4th node
echo "Scale out by adding a fourth node"
run_cmd kubectl apply -f $dir/scale_out.yaml
sleep 2
run_cmd kubectl get pods

# Perform an online rolling upgrade
echo "Perform a zero downtime upgrade of CockroachDB (note the version in the DB Console UI)"
run_cmd kubectl apply -f $dir/rolling_upgrade.yaml
echo "Check the DB Console to verify the version has changed"
echo

# Tear it down
echo
echo
echo "** Finally: tear it all down.  CAREFUL -- BE SURE YOU'RE DONE! **"
echo "Press ENTER to confirm you want to TEAR IT DOWN."
read

echo "Deleting the CRDB Embeddings app"
kubectl delete -f $dir/crdb-embeddings.yaml

echo "Deleting the SQL client"
kubectl delete -f $SQL_CLIENT_YAML

echo "Deleting the CockroachDB cluster"
kubectl delete -f $dir/cockroachdb.yaml

echo "Deleting the persistent volumes and persistent volume claims"
kubectl delete pv,pvc --all

echo "Deleting the K8s operator"
kubectl delete -f $OPERATOR_YAML

echo "Deleting the GKE cluster"
run_cmd gcloud container clusters delete $NAME --region=$REGION --quiet

