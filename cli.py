import argparse
import subprocess
import logging
import neuroglancer
import os
import sys
import signal

import select

logging.basicConfig()

processes = []
job_ids = []
hosts = []
security = "http"
import subprocess

neuroglancer.set_server_bind_address("0.0.0.0")


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


def cleanup(signum, frame):
    print(f"Script is being killed. Received signal: {signum}")

    if is_bsub_available():
        # Run your command here
        for job_id in job_ids:
            print(f"Killing job {job_id}")
            os.system(f"bkill {job_id}")
    else:
        for process in processes:
            process.kill()
    sys.exit(0)


# Attach signal handlers
signal.signal(signal.SIGINT, cleanup)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, cleanup)  # Handle termination


def get_host_from_stdout(output):
    # Print or parse the output line-by-line
    if "Host name: " in output and f"Listening at: {security}://" in output:
        host_name = output.split("Host name: ")[1].strip()
        port = output.split(f"Listening at: {security}://0.0.0.0:")[1].split(" (")[0]

        hosts.append(f"{security}://{host_name}:{port}")
        print(f"{hosts=}")
        return True
    return False


def parse_bpeek_output(job_id):
    # Run bpeek to get the job's real-time output
    command = f"bpeek {job_id}"

    try:
        # Process the output in real-time
        while True:
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
                break  # End of output
            if output:
                # Print or parse the output line-by-line
                if get_host_from_stdout(output):
                    break
                # Example: Parse a specific pattern (e.g., errors or warnings)
                if "error" in output.lower():
                    print(f"Error found: {output.strip()}")

        # Capture any error output
        error_output = process.stderr.read()
        if error_output:
            print(f"Error: {error_output.strip()}")

    except Exception as e:
        print(f"Error while executing bpeek: {e}")


def submit_bsub_job(
    job_name="my_job",
):
    if security == "https":
        command = "pixi run gunicorn --certfile=host.cert --keyfile=host.key --bind 0.0.0.0:0 --workers 1 --threads 1 example_virtual_n5_generic:app"
    else:
        command = "pixi run gunicorn --bind 0.0.0.0:0 --workers 1 --threads 1 example_virtual_n5_generic:app"
    # Get the current Conda environment
    current_directory = os.getcwd()
    # Create the bsub command
    bsub_command = [
        "bsub",
        "-J",
        job_name,
        # "-o",
        # "/dev/stdout",
        # "-e",
        # "/dev/stderr",
        "-P",
        "cellmap",
        "-q",
        "gpu_h100",
        "-gpu",
        "num=1",
        "bash",
        "-c",
        f"cd {current_directory} && hostname && pwd && {command}",
    ]

    # Submit the job
    print("Submitting job with the following command:")

    try:
        result = subprocess.run(
            bsub_command, capture_output=True, text=True, check=True
        )
        print("Job submitted successfully:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error submitting job:")
        print(e.stderr)

    return result


def generate_neuroglancer_link(dataset_path, inference_dict):
    # Create a new viewer
    viewer = neuroglancer.UnsynchronizedViewer()

    # Add a layer to the viewer
    with viewer.txn() as s:
        if ".zarr" in dataset_path:
            filetype = "zarr"
        elif ".n5" in dataset_path:
            filetype = "n5"
        else:
            filetype = "precomputed"
        if dataset_path.startswith("/"):
            dataset_path = dataset_path.replace("/nrs/cellmap/", "/nrs/").replace(
                "/groups/cellmap/cellmap/", "/dm11/"
            )
            s.layers["raw"] = neuroglancer.ImageLayer(
                source=f"{filetype}://https://cellmap-vm1.int.janelia.org/{dataset_path}",
            )
        else:
            s.layers["raw"] = neuroglancer.ImageLayer(
                source=f"{filetype}://{dataset_path}",
            )

        for host, model in inference_dict.items():
            s.layers[model] = neuroglancer.ImageLayer(source=f"n5://{host}/{model}")
        print(viewer)  # neuroglancer.to_url(viewer.state))
        while True:
            pass


def run_locally():
    # Command to execute
    command = [
        "pixi",
        "run",
        "gunicorn",
        # "--certfile=host.cert",
        # "--keyfile=host.key",
        "--bind",
        "0.0.0.0:0",
        "--workers",
        "1",
        "--threads",
        "1",
        "example_virtual_n5_generic:app",
    ]

    # Start the subprocess
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Use select to check for output without blocking
    # NOTE: For some reason the output is considered stderr here but is stdout ion cluster
    # try:
    output = ""
    while True:
        # Check if there is data available to read from stdout and stderr
        rlist, _, _ = select.select(
            [process.stdout, process.stderr], [], [], 0.1
        )  # Timeout is 0.1s

        # Read from stdout if data is available
        if process.stdout in rlist:
            output += process.stdout.readline()
        if get_host_from_stdout(output):
            break

        # Read from stderr if data is available
        if process.stderr in rlist:
            output += process.stderr.readline()
        if get_host_from_stdout(output):
            break
        # Check if the process has finished and no more output is available
        if process.poll() is not None and not rlist:
            break
    processes.append(process)
    # except KeyboardInterrupt:
    #     print("Process interrupted.")
    # finally:
    #     process.stdout.close()
    #     process.stderr.close()
    #     process.wait()  # Wait for the process to terminate


def start_hosts(num_hosts=1):
    if is_bsub_available():
        for _ in range(num_hosts):
            result = submit_bsub_job(job_name="example_job")
            job_ids.append(result.stdout.split()[1][1:-1])
        for job_id in job_ids:
            parse_bpeek_output(job_id)
    else:
        run_locally()


if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        help="Data path, including scale",
        required=True,
    )
    parser.add_argument(
        "-s", "--scale", type=str, help="Scale to apply models", required=True
    )
    parser.add_argument(
        "-m", "--models", type=str, help="Comma-separated list of models", required=True
    )

    args = parser.parse_args()
    models = args.models.split(",")
    scale = args.scale
    dataset_path = args.dataset_path
    if dataset_path.endswith("/"):
        dataset_path = dataset_path[:-1]
    print(f"Dataset: {dataset_path}, Scale: {scale}, Models: {models}")
    logging.info("Starting hosts...")
    print("starting_hosts")
    start_hosts(num_hosts=len(models))
    print("started hosts")

    logging.info("Starting hosts completed!")
    inference_dict = {}
    if len(hosts) != len(models):
        raise ValueError(
            "Number of hosts and models should be the same, but something went wrong"
        )

    print(hosts, models)
    for host, model in zip(hosts, models):
        inference_dict[host] = f"{dataset_path}__{scale}__{model}"

    generate_neuroglancer_link(dataset_path, inference_dict)
# %%
# import neuroglancer
# import time

# neuroglancer.set_server_bind_address("0.0.0.0")

viewer = neuroglancer.UnsynchronizedViewer()
with viewer.txn() as s:
    s.layers["raw"] = neuroglancer.ImageLayer(
        source="precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg"
    )
    # s.layers["raw2"] = neuroglancer.LocalVolume()
    print(viewer)
    while True:
        pass
# %%
