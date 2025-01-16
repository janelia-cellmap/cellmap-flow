import subprocess
import logging
import neuroglancer
import os
import sys
import signal
import select
import itertools
import click

logging.basicConfig()

logger = logging.getLogger(__name__)

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
    logger.error(f"Output: {output}")

    # Print or parse the output line-by-line
    if "Host name: " in output and f"* Running on {security}://" in output:
        print("Host found!")
        host_name = output.split("Host name: ")[1].split("\n")[0].strip()
        port = output.split(f"* Running on {security}://127.0.0.1:")[1].split("\n")[0]

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
            # logger.error(f"Output: {output} {error_output}")
            if (
                output == ""
                and process.poll() is not None
                and f"Job <{job_id}> : Not yet started." not in error_output
            ):
                logger.error(f"Job <{job_id}> has finished.")
                break  # End of output
            if output:
                # Print or parse the output line-by-line
                # logger.error(output)
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
    sc,
    job_name="my_job",
):
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
        sc,
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


def generate_neuroglancer_link(dataset_path, inference_dict, output_channels):
    # Create a new viewer
    viewer = neuroglancer.UnsynchronizedViewer()

    # Add a layer to the viewer
    with viewer.txn() as s:
        # if multiscale dataset
        if (
            dataset_path.split("/")[-1].startswith("s")
            and dataset_path.split("/")[-1][1:].isdigit()
        ):
            dataset_path = dataset_path.rsplit("/", 1)[0]
        if ".zarr" in dataset_path:
            filetype = "zarr"
        elif ".n5" in dataset_path:
            filetype = "n5"
        else:
            filetype = "precomputed"
        if dataset_path.startswith("/"):
            dataset_path = dataset_path.replace("/nrs/cellmap/", "nrs/").replace(
                "/groups/cellmap/cellmap/", "dm11/"
            )
            s.layers["raw"] = neuroglancer.ImageLayer(
                source=f"{filetype}://{security}://cellmap-vm1.int.janelia.org/{dataset_path}",
            )
        else:
            s.layers["raw"] = neuroglancer.ImageLayer(
                source=f"{filetype}://{dataset_path}",
            )
        colors = [
            "red",
            "green",
            "blue",
            "yellow",
            "purple",
            "orange",
            "cyan",
            "magenta",
        ]
        color_cycle = itertools.cycle(colors)
        for host, model in inference_dict.items():
            color = next(color_cycle)
            s.layers[model] = neuroglancer.ImageLayer(
                source=f"n5://{host}/{model}",
                shader=f"""#uicontrol invlerp normalized(range=[0, 255], window=[0, 255]);
#uicontrol vec3 color color(default="{color}");
void main(){{emitRGB(color * normalized());}}""",
            )
        print(viewer)  # neuroglancer.to_url(viewer.state))
        logger.error(f"link : {viewer}")
        while True:
            pass


def run_locally(sc):
    # Command to execute
    command = sc.split(" ")

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


def start_hosts(dataset, script_path, num_hosts=1):
    if security == "https":
        sc = f"cellmap_flow_server -d {dataset} -c {script_path} --certfile=host.cert --keyfile=host.key"
    else:
        sc = f"cellmap_flow_server -d {dataset} -c {script_path}"

    if is_bsub_available():
        for _ in range(num_hosts):
            result = submit_bsub_job(sc, job_name="example_job")
            job_ids.append(result.stdout.split()[1][1:-1])
        for job_id in job_ids:
            parse_bpeek_output(job_id)
    else:
        run_locally(sc)


@click.command()
@click.option(
    "-d",
    "--dataset_path",
    type=str,
    help="Data path, including scale",
    required=True,
)
@click.option(
    "-c", "--code", type=str, help="Path to the script to run", required=False
)
@click.option(
    "-ch",
    "--output_channels",
    type=int,
    help="Number of output channels",
    required=False,
    default=0,
)
def main(dataset_path, code, output_channels):

    if dataset_path.endswith("/"):
        dataset_path = dataset_path[:-1]
    # print(f"Dataset: {dataset_path}, Scale: {scale}, Models: {models}")
    logging.info("Starting hosts...")
    start_hosts(dataset_path, code, num_hosts=1)

    logging.info("Starting hosts completed!")
    inference_dict = {}
    models = ["model"]
    if len(hosts) != len(models):
        raise ValueError(
            "Number of hosts and models should be the same, but something went wrong"
        )

    # print(hosts, models)
    for host, model in zip(hosts, models):
        inference_dict[host] = f"{dataset_path}"

    generate_neuroglancer_link(dataset_path, inference_dict, output_channels)


if __name__ == "__main__":
    main()
