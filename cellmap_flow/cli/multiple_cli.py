import sys
from cellmap_flow.utils.data import DaCapoModelConfig, BioModelConfig, ScriptModelConfig
import logging
from cellmap_flow.utils.bsub_utils import start_hosts
from cellmap_flow.utils.neuroglancer_utils import generate_neuroglancer_link


data_args = ["-d", "--data-path"]
charge_back_arg = ["-P", "--project"]
server_queue_arg = ["-q", "--queue"]

DEFAULT_SERVER_QUEUE = "gpu_h100"

SERVER_COMMAND = "cellmap_flow_server"

logger = logging.getLogger(__name__)


def main():
    """
    Allows chaining multiple model calls in one command, for example:

    \b
       cellmap_flow_multiple --data-path /some/shared/path --dacapo -r run_1 -it 60 --dacapo -r run_2 -it 50 --script -s /path/to/script

    This will parse the arguments and dispatch the appropriate logic
    for each sub-command (dacapo, script, etc.).
    """

    args = sys.argv[1:]

    if not args:
        logger.error("No arguments provided.")
        sys.exit(1)

    if data_args[0] not in args and data_args[1] not in args:
        logger.error("Missing required argument: --data-path")
        sys.exit(1)

    if charge_back_arg[0] not in args and charge_back_arg[1] not in args:
        logger.error("Missing required argument: --project")
        sys.exit(1)

    if server_queue_arg[0] not in args and server_queue_arg[1] not in args:
        logger.warning(
            f"Missing required argument: --queue, using default queue {DEFAULT_SERVER_QUEUE}"
        )
        args.extend([server_queue_arg[0], DEFAULT_SERVER_QUEUE])

    if "--dacapo" not in args and "--script" not in args and "--bioimage" not in args:
        logger.error(
            "Missing required argument at least one should exist: --dacapo, --script, or --bioimage"
        )
        logger.error(
            "Example: cellmap_flow_multiple --data-path /some/shared/path --dacapo -r run_1 -it 60 --dacapo -r run_2 -it 50 --script -s /path/to/script"
        )
        logger.error("Now we will just open the raw data ..")

    # Extract data path
    data_path = None
    charge_back = None
    queue = None
    models = []

    for i, arg in enumerate(args):
        if arg in charge_back_arg:
            if charge_back is not None:
                logger.error("Multiple charge back projects provided.")
                sys.exit(1)
            charge_back = args[i + 1]
        if arg in server_queue_arg:
            if queue is not None:
                logger.error("Multiple server queues provided.")
                sys.exit(1)
            queue = args[i + 1]

        if arg in data_args:
            if data_path is not None:
                logger.error("Multiple data paths provided.")
                sys.exit(1)
            data_path = args[i + 1]

    if not data_path:
        logger.error("Data path not provided.")
        sys.exit(1)

    if not charge_back:
        logger.error("Charge back project not provided.")
        sys.exit(1)

    if not queue:
        logger.error("Server queue not provided.")
        sys.exit(1)

    print("Data path:", data_path)

    i = 0
    while i < len(args):
        token = args[i]

        if token == "--dacapo":
            # We expect: --dacapo -r run_name -it iteration -n "some name"
            run_name = None
            iteration = 0
            name = None

            j = i + 1
            while j < len(args) and not args[j].startswith("--"):
                if args[j] in ("-r", "--run-name"):
                    run_name = args[j + 1]
                    j += 2
                elif args[j] in ("-i", "--iteration"):
                    iteration = int(args[j + 1])
                    j += 2
                elif args[j] in ("-n", "--name"):
                    name = args[j + 1]
                    j += 2
                else:
                    j += 1

            if not run_name:
                logger.error("Missing -r/--run-name for --dacapo sub-command.")
                sys.exit(1)

            models.append(DaCapoModelConfig(run_name, iteration, name=name))
            i = j
            continue

        elif token == "--script":
            # We expect: --script -s script_path -n "some name"
            script_path = None
            name = None

            j = i + 1
            while j < len(args) and not args[j].startswith("--"):
                if args[j] in ("-s", "--script_path"):
                    script_path = args[j + 1]
                    j += 2
                elif args[j] in ("-n", "--name"):
                    name = args[j + 1]
                    j += 2
                else:
                    j += 1

            if not script_path:
                logger.error("Missing -s/--script_path for --script sub-command.")
                sys.exit(1)

            models.append(ScriptModelConfig(script_path, name=name))
            i = j
            continue

        elif token == "--bioimage":
            # We expect: --bioimage -m model_path -n "some name"
            model_path = None
            name = None

            j = i + 1
            while j < len(args) and not args[j].startswith("--"):
                if args[j] in ("-m", "--model_path"):
                    model_path = args[j + 1]
                    j += 2
                elif args[j] in ("-n", "--name"):
                    name = args[j + 1]
                    j += 2
                else:
                    j += 1

            if not model_path:
                logger.error("Missing -m/--model_path for --bioimage sub-command.")
                sys.exit(1)

            models.append(BioModelConfig(model_path, name=name))
            i = j
            continue

        else:
            # If we don't recognize the token, just move on
            i += 1

    # Print out the model configs for debugging
    for model in models:
        print(model)

    run_multiple(models, data_path, charge_back, queue)


if __name__ == "__main__":
    main()


def run_multiple(models, dataset_path, charge_back, queue):
    inference_dict = {}
    for model in models:
        command = f"{SERVER_COMMAND} {model.command} -d {dataset_path}"
        host = start_hosts(
            command, job_name=model.name, queue=queue, charge_group=charge_back
        )
        if host is None:
            raise Exception("Could not start host")
        inference_dict[host] = model.name
    generate_neuroglancer_link(dataset_path, inference_dict)
