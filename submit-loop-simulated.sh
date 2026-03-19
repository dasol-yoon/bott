#!/bin/sh

for trial in $(seq 1 1 20)
do
    for algo in EICF KG EI Random
    do
        for num_iter in 50
        do
            for param_truth in "380 1.5 -1.5" 
            do
                for n_init_evals in 7
                do
                    for noisy_ground_truth_peak in 0
                    do
                    sbatch -J run0_${trial}_${algo}_${noisy_ground_truth_peak} \
                        -o ./logs/Tri_${trial}_Alg_${algo}_noisy_${noisy_ground_truth_peak}_%j.out \
                            -e ./logs/Tri_${trial}_Alg_${algo}_noisy_${noisy_ground_truth_peak}_%j.err \
                            --requeue simulated.sub ${trial} ${algo} ${num_iter} "$param_truth" ${n_init_evals} ${noisy_ground_truth_peak}
                            sleep 0.1s
                        done
                done
            done
        done
    done
done