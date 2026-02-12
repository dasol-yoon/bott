#!/bin/sh

for trial in $(seq 18 1 18)
do
    for algo in EI
    do
        for num_iter in 50
        do
            for n_init_evals in 7
            do
                sbatch -J STO_${trial}_${algo} \
                    -o ./logs/${trial}_${algo}_%j.out \
                        -e ./logs/${trial}_${algo}_%j.err \
                        --requeue STO28.sub ${trial} ${algo} ${num_iter} ${n_init_evals}
                    sleep 0.1s
            done
        done
    done
done