import os

with open(os.path.join('logs','RandomForest_2020_6_26.log')) as log_info:
    info = log_info.readlines()
    run_time = [ run.split('is ')[1] for run in info if 'Run Time' in run]
    minutes = [  int(run.split('.')[0]) for run in run_time ]

print(f"The run time {sum(minutes)} minutes or {sum(minutes) // 60} hours and {sum(minutes) % 60} minutes")