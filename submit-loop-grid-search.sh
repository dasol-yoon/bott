#!/bin/sh

mkdir -p ./logs

gt_job=$(sbatch --parsable \
    -J grid_search_gt \
    -o ./logs/grid_search_gt_%j.out \
    -e ./logs/grid_search_gt_%j.err \
    --requeue grid_search.sub gt)
echo "Submitted ground-truth job $gt_job"

for job_id in $(seq 0 49)
do
    sbatch --dependency=afterok:${gt_job} \
        -J grid_search_${job_id} \
        -o ./logs/grid_search_job_${job_id}_%j.out \
        -e ./logs/grid_search_job_${job_id}_%j.err \
        --requeue grid_search.sub ${job_id}
    sleep 0.1s
done

echo "After all 50 jobs finish, combine them with:"
echo "  /home/pb482/miniconda3/envs/bott/bin/python /home/pb482/bott/runner_file/grid_search_combine.py"
