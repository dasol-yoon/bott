#!/bin/sh

for trial in $(seq 1 1 1)
do
    for algo in EICF 
    do
        for num_iter in 50
        do
            for param_truth in "380 1.5 -1.5" 
            do
                for n_init_evals in 7
                do
                    sbatch -J run0_${trial}_${algo} \
                        -o ./logs/${trial}_${algo}_%j.out \
                            -e ./logs/${trial}_${algo}_%j.err \
                            --requeue simulated.sub ${trial} ${algo} ${num_iter} "$param_truth" ${n_init_evals}
                        sleep 0.1s
                done
            done
        done
    done
done