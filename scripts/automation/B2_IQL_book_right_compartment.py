import os

def main():
    num_job_group = 1
    sh_command = './run.sh'
    pre_sbatch_command = 'MUJOCO_GL=egl WANDB__SERVICE_WAIT=86400 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 WANDB_API_KEY=<TOKEN HERE!!!!!>'
    num_groups = 4
    num_cpus = 1
    sbatch_command = f'-A co_rail -p savio4_gpu --gres=gpu:A5000:1 -N 1 -n {num_groups} -c {num_cpus} --qos=rail_gpu4_normal -t 15:00:00 --mem=60G --requeue'
    python_command = 'python3 main.py'

    run_group = os.path.splitext(os.path.basename(__file__))[0]

    print(run_group)

    default_args = dict(
        run_group=run_group,
        eval_interval=100000,
        save_interval=10000000,
        log_interval=10000,
        eval_episodes=50,
        video_episodes=1,
    )

    tests = []
    group_num = int(run_group[1:].split('_')[0])
    seed = group_num * 10000
    print('seed', seed)

    for agent in ['agents/crl/original.py']:
        for offline_steps in [1000000]:
            for env_name, alphas in [
                ('pointmaze-medium-navigate-v0', [0.03]),
                ('pointmaze-large-navigate-v0', [0.03]),
                ('antmaze-medium-navigate-v0', [0.1]),
                ('antmaze-large-navigate-v0', [0.1]),
                ('antmaze-giant-navigate-v0', [0.1]),
                ('humanoidmaze-medium-navigate-v0', [0.1]),
                ('humanoidmaze-large-navigate-v0', [0.1]),
                ('antsoccer-arena-navigate-v0', [0.3]),
                ('cube-single-play-v0', [3.0]),
                ('cube-double-play-v0', [3.0]),
                ('scene-play-v0', [3.0]),
                ('puzzle-3x3-play-v0', [3.0]),
                ('puzzle-4x4-play-v0', [3.0]),
            ]:
                for alpha in alphas:
                    for norm in [False]:
                        for discount in [0.995] if ('humanoid' in env_name or 'giant' in env_name) else [0.99]:
                            for i in range(4):
                                seed += 1
                                base_dict = dict(
                                    default_args,
                                )
                                tests.append(dict(
                                    base_dict,
                                    seed=seed,
                                    env_name=env_name,
                                    train_steps=offline_steps,
                                    agent=agent,
                                    agentIbatch_size=1024,
                                    agentIdiscount=discount,
                                    agentIalpha=alpha,
                                    eval_episodes=50,
                                    eval_on_cpu=0,
                                    agentIoraclerep=False,
                                    agentInorm=norm
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
    with open('sbatch.sh', 'w') as f:
        f.write('\n'.join(contents))

if __name__ == '__main__':
    main()