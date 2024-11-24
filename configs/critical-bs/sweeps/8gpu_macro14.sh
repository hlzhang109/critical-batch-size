#!/bin/bash

# Initial port range start
PORT_START=10000
PORT_END=40000
MASTER_PORT=$PORT_START

# Function to check if the port is in use
is_port_in_use() {
    netstat -tuln | grep $1 > /dev/null
    return $?
}

# Find an unused port starting from PORT_START
while [ $MASTER_PORT -le $PORT_END ]; do
    is_port_in_use $MASTER_PORT
    if [ $? -eq 1 ]; then
        echo "Found available port: $MASTER_PORT"
        break
    else
        MASTER_PORT=$((MASTER_PORT+1))
    fi
done

if [ $MASTER_PORT -gt $PORT_END ]; then
    echo "Error: Could not find an available port in the range $PORT_START-$PORT_END."
    exit 1
fi

wandb online
# Set the MASTER_PORT environment variable
export MASTER_PORT

export WOLRD_SIZE=8
# export MASTER_PORT=$((RANDOM % 10001 + 10000))
export OMP_NUM_THREADS=23 # total 96 cores
echo $OMP_NUM_THREADS
torchrun --master-port=$MASTER_PORT --nnodes 1 --nproc_per_node=$WOLRD_SIZE $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14}