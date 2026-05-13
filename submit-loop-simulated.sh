#!/bin/sh

for trial in $(seq 1 1 20)
do
    for algo in EI Random EICF KG
    do
        for num_iter in 50
        do
            for param_truth in "100 1.5 -1.5" 
            do
                for n_init_evals in 7
                do
                    for noisy_ground_truth_peak in 0
                    do
                    sbatch -J simulated_trial_${trial}_algo_${algo}_noisy_${noisy_ground_truth_peak} \
                        -o ./logs/simulated_trial_${trial}_algo_${algo}_noisy_${noisy_ground_truth_peak}_%j.out \
                            -e ./logs/simulated_trial_${trial}_algo_${algo}_noisy_${noisy_ground_truth_peak}_%j.err \
                            --requeue simulated.sub ${trial} ${algo} ${num_iter} "$param_truth" ${n_init_evals} ${noisy_ground_truth_peak}
                            sleep 0.1s
                        done
                done
            done
        done
    done
done