import os
os.system("tensor_python dist_keras.py --job_name=ps --ps_hosts=172.22.191.46:2225 --worker_hosts=172.22.191.46:2226 --task_index=0")
