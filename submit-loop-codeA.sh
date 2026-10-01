#!/bin/sh

for trial in $(seq 3 1 20)
do
    for algo in EICF EI KG Random
    do
        for num_iter in 50
        do
            for n_init_evals in 7
            do
                if [ "$algo" = "EICF" ]; then
                    for eps_c in 1 2 5
                    do
                        sbatch -J codeA_${trial}_${algo}_epsc_${eps_c} \
                            -o ./logs/codeA_Tri_${trial}_${algo}_epsc_${eps_c}_%j.out \
                                -e ./logs/codeA_Tri_${trial}_${algo}_epsc_${eps_c}_%j.err \
                                --requeue code_A.sub ${trial} ${algo} ${num_iter} ${n_init_evals} ${eps_c}
                            sleep 0.1s
                    done
                else
                    sbatch -J codeA_${trial}_${algo} \
                        -o ./logs/codeA_Tri_${trial}_${algo}_%j.out \
                            -e ./logs/codeA_Tri_${trial}_${algo}_%j.err \
                            --requeue code_A.sub ${trial} ${algo} ${num_iter} ${n_init_evals}
                        sleep 0.1s
                fi
            done
        done
    done
done
