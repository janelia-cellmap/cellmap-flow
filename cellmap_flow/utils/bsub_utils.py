import subprocess
import logging
import os
import sys
import signal
import select
from cellmap_flow.globals import g
import logging


from cellmap_flow.utils.web_utils import IP_PATTERN

logger = logging.getLogger(__name__)

security = "http"

SERVER_COMMAND = "cellmap_flow_server"


class Job:
    def __init__(
        self, job_id=None, model_name=None, status="running", host=None, process=None
    ):
        self.job_id = job_id
        self.model_name = model_name
        self.status = status
        self.host = host
        self.process = process

    def kill(self):
        if self.job_id is None and self.process is None:
            logger.error("Job is not running.")
            return
        if self.process is not None:
            print(f"Killing process {self.process.pid}")
            self.process.kill()
            self.process = None
            self.status = "killed"
        if self.job_id is not None:
            self.status = "killed"
            os.system(f"bkill {self.job_id}")


def cleanup(signum, frame):
    print(f"Script is being killed. Received signal: {signum}")
    for job in g.jobs:
        print(f"Killing job {job.job_id}")
        job.kill()

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
    queue="gpu_h100",
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


def get_host_from_stdout(output):
    # parts = IP_PATTERN.split("ip_address")

    if IP_PATTERN[0] in output and IP_PATTERN[1] in output:
        host = output.split(IP_PATTERN[0])[1].split(IP_PATTERN[1])[0]
        return host
    return None


# def get_host_from_stdout(output):

#     if "Host name: " in output and f"* Running on {security}://" in output:
#         host_name = output.split("Host name: ")[1].split("\n")[0].strip()
#         port = output.split(f"* Running on {security}://127.0.0.1:")[1].split("\n")[0]


#         host = f"{security}://{host_name}:{port}"
#         print(f"{host}")
#         return host
#     return None


def run_locally(sc, name):
    command = sc.split(" ")
    print(f"Running command: {command}")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    output = ""
    while True:
        # print("Waiting for output...")
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
        # print(output)

        host = get_host_from_stdout(output)
        if host:
            break
        # Check if the process has finished and no more output is available
        if process.poll() is not None and not rlist:
            break
    g.jobs.append(Job(model_name=name, host=host, process=process))
    return host


def start_hosts(
    command, queue="gpu_h100", charge_group="cellmap", job_name="example_job"
):
    g.queue = queue
    g.charge_group = charge_group
    if security == "https":
        command = f"{command} --certfile=host.cert --keyfile=host.key"

    if is_bsub_available():
        result = submit_bsub_job(
            command, queue, charge_group, job_name=f"{job_name}_server"
        )
        job_id = result.stdout.split()[1][1:-1]
        host = parse_bpeek_output(job_id)
        new_job = Job(job_id=job_id, model_name=job_name, host=host)
        g.jobs.append(new_job)
    else:
        new_job = run_locally(command, job_name)

    return new_job
