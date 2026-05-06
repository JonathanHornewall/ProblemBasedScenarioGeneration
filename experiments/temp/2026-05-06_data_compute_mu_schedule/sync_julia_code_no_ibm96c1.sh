#!/bin/bash -e

for server in gcp-4c-{2,3}
do
    rsync -avu \
        --delete \
        --exclude='*.csv' \
        --exclude='*.log' \
        --exclude='sandbox/**.jls' \
        --exclude='sandbox/**.csv' \
        --exclude='src/ContextualDFL/ContextualDFLTraining/results' \
        --exclude='src/ContextualDFL/ContextualDFLExperiments/experiments/resource_allocation_annealing/results' \
        --exclude='.git' \
        ~/ProblemBasedScenarioGeneration \
        "${server}": &
done
wait
echo "Syncing done"
