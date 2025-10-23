import os
import numpy as np

def main():
    num_job_group = 1
    sh_command = 'scripts/automatic/run.sh'
    wandb_env_variables = 'WANDB_SERVICE_WAIT=86400 WANDB_NETWORK_TIMEOUT=600 WANDB_FILE_TRANSFER_TIMEOUT=1200 WANDB_INIT_TIMEOUT=300 WANDB_HTTP_TIMEOUT=600 WANDB_RETRY_ATTEMPTS=15 WANDB_RETRY_WAIT_MIN=5 WANDB_RETRY_WAIT_MAX=120 '
    pre_sbatch_command = f'{wandb_env_variables} MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '
    num_groups = 1
    num_cpus = 8
    qos = 'high'
    sbatch_command = f'-A co_rail -p savio4_gpu --gres=gpu:A5000:1 -N 1 -n {num_groups} -c {num_cpus} --qos=rail_gpu4_{qos} -t 12:00:00 --mem=60G --requeue '
    python_command = 'uv run main.py '

    run_group = os.path.splitext(os.path.basename(__file__))[0]

    print(run_group)

    default_args = dict(
        run_group=run_group,
        online_steps=0,
        eval_interval=100_000,
        num_parallel_envs=5, # can't go too crazy with this, brc crashes otherwise
        save_interval=-1,
        log_interval=5_000,
        eval_episodes=50,
        video_episodes=5,
        horizon_length=5,
        env_name='libero_90-study_scene1-pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy'
    )

    ## things to sweep over:  agent.num_qs, agent.flow_steps, agent.normalize_q_loss,gent.actor_num_samples, agent.use_fourier_features, agent.weight_decay

    tests = []
    group_num = int(run_group[1:].split('_')[0])
    seed = group_num * 10000
    print('seed', seed)

    for agent in ['agents/acfql.py']:
        for num_qs in [2]:
            for flow_steps in [10]:
                 for normalize_q_loss in [False]:
                    for actor_num_samples in [4, 8, 16, 32, 64]:
                        for use_fourier_features in [False]:
                            for weight_decay in [0.0]:
                                for i in range(4):
                                    seed += 1
                                    base_dict = dict(
                                        default_args,
                                    )
                                    tests.append(dict(
                                        base_dict,
                                        seed=seed,
                                        agent=agent,
                                        agentIbatch_size=256,
                                        agentIactor_type='best-of-n',
                                        agentInum_qs=num_qs,
                                        agentIflow_steps=flow_steps,
                                        agentInormalize_q_loss=normalize_q_loss,
                                        agentIactor_num_samples=actor_num_samples,
                                        agentIuse_fourier_features=use_fourier_features,
                                        agentIweight_decay=weight_decay,
                                    ))

    print(len(tests))

    test_commands = []
    for test in tests:
        test_command = ''
        for k, v in test.items():
            if v is None:
                continue
            test_command += f' --{k.replace("I", ".")}={v}'
        test_commands.append(test_command)

    print(f"Sample test command: {python_command}{test_commands[np.random.randint(len(test_commands))]}")

    contents = []
    content = ''
    target_remainder = num_groups - 1
    for i, test_command in enumerate(test_commands):
        if i % num_groups == 0:
            content += f'{pre_sbatch_command} sbatch {sbatch_command} --parsable --comment="{run_group}.{i // num_groups}" {sh_command}'
            if i + num_groups >= len(test_commands):
                target_remainder = len(test_commands) - i - 1
        content += f" '{python_command}{test_command}'"
        if i % num_groups == target_remainder:
            contents.append(content)
            content = ''
    if num_job_group is not None:
        for i, content in enumerate(contents):
            contents[i] = f'jobid{i}=$({content}) && echo $jobid{i}'
        for i, content in enumerate(contents):
            if i % num_job_group != 0:
                cur = content.split('sbatch')
                cur[1] = f' --dependency=afterany:$jobid{i - 1}' + cur[1]
                contents[i] = 'sbatch'.join(cur)
    with open(f'scripts/shell_scripts/{run_group}_sbatch.sh', 'w') as f:
        f.write('\n'.join(contents))
    os.chmod(f'scripts/shell_scripts/{run_group}_sbatch.sh', 0o755) # give execute permission
    print(f'./scripts/shell_scripts/{run_group}_sbatch.sh created. Simply run this file!')

if __name__ == '__main__':
    main()
