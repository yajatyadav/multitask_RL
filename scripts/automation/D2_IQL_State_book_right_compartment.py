import os
import numpy as np

def main():
    num_job_group = 1
    sh_command = 'scripts/automation/run.sh'
    pre_sbatch_command = 'MUJOCO_GL=egl WANDB__SERVICE_WAIT=86400 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '
    num_groups = 4
    num_cpus = 1
    qos = 'high'
    sbatch_command = f'-A co_rail -p savio4_gpu --gres=gpu:A5000:1 -N 1 -n {num_groups} -c {num_cpus} --qos=rail_gpu4_{qos} -t 24:00:00 --mem=60G --requeue '
    python_command = 'uv run scripts/train_offline.py '

    run_group = os.path.splitext(os.path.basename(__file__))[0]

    print(run_group)

    default_args = dict(
        batch_size=256,
        run_group=run_group,
        pixel_observations=False,
        eval_interval=50_000,
        save_interval=1_000_000,
        log_interval=5_000,
        eval_episodes=50,
        video_episodes=5,
    )

    tests = []
    group_num = int(run_group[1:].split('_')[0])
    seed = group_num * 10000
    print('seed', seed)

    for agent in ['agents/iql.py']:
        for offline_steps in [1_000_000]:
            for task_suite_name, task_name, train_dataset_mix_name in [
                ('libero_90', 'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy', 'libero_90__book_right_compartment_caddy__study_scene1'),
            ]:
                for alpha in [0.0, 0.1, 0.3, 1, 3, 10]:
                    for expectile in [0.7, 0.8, 0.9]:
                        for value_layer_norm in [True]:
                            for state_dependent_std in [False]:
                                for i in range(4):
                                    seed += 1
                                    base_dict = dict(
                                        default_args,
                                    )
                                    tests.append(dict(
                                        base_dict,
                                        seed=seed,
                                        task_suite_name=task_suite_name,
                                        task_name=task_name,
                                        train_dataset_mix_name=train_dataset_mix_name,
                                        offline_steps=offline_steps,
                                        agent=agent,
                                        agentIstate_dependent_std=state_dependent_std,
                                        agentIexpectile=expectile,
                                        agentIalpha=alpha,
                                        agentIlayer_norm=value_layer_norm,
                                        agentIencoder="state_space_encoder",
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