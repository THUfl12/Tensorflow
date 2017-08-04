import os
os.system("tensor_python dist_keras.py --job_name=worker --ps_hosts=172.22.191.46:2225 --worker_hosts=172.22.191.46:2226,172.22.191.45:2225 --task_index=1")
