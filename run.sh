#!/bin/bash

#PJM -L rscgrp=cx-single
#PJM -L gpu=4
#PJM -L elapse=336:00:00
#PJM -L jobenv=singularity
#PJM -j

module load singularity
singularity exec \
        --bind $HOME,/data/group1/${USER} \
        --nv /data/group1/${USER}/latest.sif \
        bash _run.sh
