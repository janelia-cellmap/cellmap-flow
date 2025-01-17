#!/usr/bin/env python3

import sys
from cellmap_flow.utils.data import DaCapoModelConfig, BioModelConfig, ScriptModelConfig
import logging
from cellmap_flow.utils.bsub_utils import is_bsub_available, submit_bsub_job, parse_bpeek_output, run_locally, start_hosts, job_ids, security
from cellmap_flow.utils.neuroglancer_utils import generate_neuroglancer_link


data_args = ["-d", "--data-path"]
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

   if "--dacapo" not in args and "--script" not in args and "--bioimage" not in args:
      logger.error("Missing required argument at least one should exist: --dacapo, --script, or --bioimage")
      sys.exit(1)

   # Extract data path
   data_path = None
   models = []

   for i, arg in enumerate(args):
      if arg in data_args:
         if data_path is not None:
            logger.error("Multiple data paths provided.")
            sys.exit(1)
         data_path = args[i + 1]

   if not data_path:
      logger.error("Data path not provided.")
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
                  run_name = args[j+1]
                  j += 2
               elif args[j] in ("-it", "--iteration"):
                  iteration = int(args[j+1])
                  j += 2
               elif args[j] in ("-n", "--name"):
                  name = args[j+1]
                  j += 2
               else:
                  j += 1

         if not run_name:
               logger.error("Missing -r/--run-name for --dacapo sub-command.")

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
                  script_path = args[j+1]
                  j += 2
               elif args[j] in ("-n", "--name"):
                  name = args[j+1]
                  j += 2
               else:
                  j += 1

         if not script_path:
               logger.error("Missing -s/--script_path for --script sub-command.")

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
                  model_path = args[j+1]
                  j += 2
               elif args[j] in ("-n", "--name"):
                  name = args[j+1]
                  j += 2
               else:
                  j += 1

         if not model_path:
               logger.error("Missing -m/--model_path for --bioimage sub-command.")

         models.append(BioModelConfig(model_path, name=name))
         i = j
         continue

      else:
         # If we don't recognize the token, just move on
         i += 1

   # Print out the model configs for debugging
   for model in models:
      print(model)

   run_multiple(models, data_path)

if __name__ == "__main__":
   main()


def run_multiple(models,dataset_path):
   inference_dict = {}
   for model in models:
      command = f"{model.command} -d {dataset_path}"
      host = start_hosts(command,job_name=model.name)
      if host is None:
         raise Exception("Could not start host")
      inference_dict[host] = model.name
   generate_neuroglancer_link(dataset_path, inference_dict)

