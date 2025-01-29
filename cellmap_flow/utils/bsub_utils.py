import subprocess
import logging
import neuroglancer
import os
import sys
import signal
import select
import itertools
import click

from cellmap_flow.utils.data import IP_PATTERN

processes = []
job_ids = []
security = "http"


def cleanup(signum, frame):
    print(f"Script is being killed. Received signal: {signum}")
    if is_bsub_available():
        for job_id in job_ids:
            print(f"Killing job {job_id}")
            os.system(f"bkill {job_id}")
    else:
        for process in processes:
            process.kill()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, cleanup)

logger = logging.getLogger(__name__)


def is_bsub_available():
    try:
        # Run 'which bsub' to check if bsub is available in PATH
        result = subprocess.run(
            ["which", "bsub"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if result.stdout:
            return True
        else:
            return False
    except Exception as e:
        print("Error:", e)


def submit_bsub_job(
    command,
    queue="gpu_a100",
    charge_group="cellmap",
    job_name="my_job",
):
    bsub_command = ["bsub", "-J", job_name]
    if charge_group:
        bsub_command += ["-P", charge_group]
    bsub_command += [
        "-q",
        queue,
        "-gpu",
        "num=1",
        "bash",
        "-c",
        command,
    ]

    print("Submitting job with the following command:")

    try:
        result = subprocess.run(
            bsub_command, capture_output=True, text=True, check=True
        )
        print("Job submitted successfully:")
        print(result.stdout)
        return result
    except Exception as e:
        print("Error submitting job:")
        print("It can be that you didn't define the charge group. -P <charge_group>")
        raise (e)


def parse_bpeek_output(job_id):
    command = f"bpeek {job_id}"
    host = None
    try:
        # Process the output in real-time
        while True:
            # logger.error(f"Running command: {command}")
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            output = process.stdout.read()
            error_output = process.stderr.read()
            if (
                output == ""
                and process.poll() is not None
                and f"Job <{job_id}> : Not yet started." not in error_output
            ):
                logger.error(f"Job <{job_id}> has finished.")
                break  # End of output
            if output:
                host = get_host_from_stdout(output)
                if host:
                    break
                if "error" in output.lower():
                    print(f"Error found: {output.strip()}")

        error_output = process.stderr.read()
        if error_output:
            print(f"Error: {error_output.strip()}")

    except Exception as e:
        print(f"Error while executing bpeek: {e}")

    return host


# def get_host_from_stdout(output):
#     parts = IP_PATTERN.split("ip_address")

#     if parts[0] in output and parts[1] in output:
#         host = output.split(parts[0])[1].split(parts[1])[0]
#         return host
#     return None


def get_host_from_stdout(output):
    if "Host name: " in output and f"* Running on {security}://" in output:
        host_name = output.split("Host name: ")[1].split("\n")[0].strip()
        port = output.split(f"* Running on {security}://127.0.0.1:")[1].split("\n")[0]

        host = f"{security}://{host_name}:{port}"
        print(f"{host}")
        return host
    return None


def run_locally(sc):
    command = sc.split(" ")

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    output = ""
    while True:
        # Check if there is data available to read from stdout and stderr
        rlist, _, _ = select.select(
            [process.stdout, process.stderr], [], [], 0.1
        )  # Timeout is 0.1s

        # Read from stdout if data is available
        if process.stdout in rlist:
            output += process.stdout.readline()
        host = get_host_from_stdout(output)
        if host:
            break

        # Read from stderr if data is available
        if process.stderr in rlist:
            output += process.stderr.readline()
        host = get_host_from_stdout(output)
        if host:
            break
        # Check if the process has finished and no more output is available
        if process.poll() is not None and not rlist:
            break
    processes.append(process)
    return host


def start_hosts(
    command, queue="gpu_h100", charge_group="cellmap", job_name="example_job"
):
    if security == "https":
        command = f"{command} --certfile=host.cert --keyfile=host.key"

    if is_bsub_available():
        result = submit_bsub_job(command, queue, charge_group, job_name=job_name)
        job_id = result.stdout.split()[1][1:-1]
        job_ids.append(job_id)
        host = parse_bpeek_output(job_id)
    else:
        host = run_locally(command)

    return host
