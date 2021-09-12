"""
Create subprocess to run multiple gym servers
"""

import subprocess
import argparse


def get_cmd(port):
    filename = 'gym_http_server.py'
    cmd = f"python {filename} -p {port}"
    return cmd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_servers', required=True)
    args = vars(parser.parse_args())
    process = []
    num_servers = int(args['num_servers'])
    for i in range(num_servers):
        p = subprocess.Popen(get_cmd(i + 5000).split())
        process.append(p)
    for p in process:
        p.wait()
