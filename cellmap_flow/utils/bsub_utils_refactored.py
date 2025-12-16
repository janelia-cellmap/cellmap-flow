"""
Utilities for job submission and management using LSF (bsub) or local execution.

This module handles:
- Job submission via bsub (LSF cluster)
- Local process execution as fallback
- Job monitoring and output parsing
- Signal handling for cleanup
"""

import subprocess
import logging
import os
import sys
import signal
import select
from typing import Optional, List
from dataclasses import dataclass

from cellmap_flow.globals import g
from cellmap_flow.utils.web_utils import IP_PATTERN

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SECURITY = "http"
DEFAULT_QUEUE = "gpu_h100"
DEFAULT_CHARGE_GROUP = "cellmap"
SERVER_COMMAND = "cellmap_flow_server"


@dataclass
class Job:
    """Represents a submitted job (either bsub or local process)."""
    
    job_id: Optional[str] = None
    model_name: Optional[str] = None
    status: str = "running"
    host: Optional[str] = None
    process: Optional[subprocess.Popen] = None

    def kill(self) -> None:
        """Kill the job (either bsub job or local process)."""
        if self.job_id is None and self.process is None:
            logger.error("Job is not running.")
            return
        
        if self.process is not None:
            logger.info(f"Killing local process {self.process.pid}")
            self.process.kill()
            self.process = None
            self.status = "killed"
        
        if self.job_id is not None:
            logger.info(f"Killing bsub job {self.job_id}")
            self.status = "killed"
            subprocess.run(["bkill", self.job_id], capture_output=True)


def cleanup_handler(signum: int, frame) -> None:
    """
    Signal handler for graceful shutdown.
    Kills all tracked jobs before exiting.
    """
    logger.warning(f"Received signal {signum}. Cleaning up jobs...")
    for job in g.jobs:
        logger.info(f"Killing job: {job.model_name} (ID: {job.job_id})")
        job.kill()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, cleanup_handler)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, cleanup_handler)  # Handle termination


def is_bsub_available() -> bool:
    """Check if bsub command is available in the system PATH."""
    try:
        result = subprocess.run(
            ["which", "bsub"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        return bool(result.stdout)
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"bsub not available: {e}")
        return False


def submit_bsub_job(
    command: str,
    queue: str = DEFAULT_QUEUE,
    charge_group: Optional[str] = None,
    job_name: str = "my_job",
    num_gpus: int = 1,
    num_cpus: int = 4,
) -> subprocess.CompletedProcess:
    """
    Submit a job to LSF cluster using bsub.
    
    Args:
        command: Shell command to execute
        queue: LSF queue name
        charge_group: Project/chargeback group for billing
        job_name: Name for the job
        num_gpus: Number of GPUs to request
        num_cpus: Number of CPUs to request
        
    Returns:
        CompletedProcess with stdout containing job ID
        
    Raises:
        subprocess.CalledProcessError: If job submission fails
    """
    bsub_command = ["bsub", "-J", job_name]
    
    if charge_group:
        bsub_command += ["-P", charge_group]
    
    bsub_command += [
        "-q", queue,
        "-gpu", f"num={num_gpus}",
        "-n", str(num_cpus),
        "bash", "-c", command,
    ]

    logger.info(f"Submitting bsub job: {' '.join(bsub_command)}")

    try:
        result = subprocess.run(
            bsub_command,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        logger.info(f"Job submitted successfully: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Job submission failed: {e.stderr}")
        if not charge_group:
            logger.error("Hint: You may need to specify a charge group with -P option")
        raise


def extract_host_from_output(output: str) -> Optional[str]:
    """
    Extract host/URL from command output using configured patterns.
    
    Args:
        output: String output to search
        
    Returns:
        Host URL if found, None otherwise
    """
    if not output:
        return None
    
    try:
        if IP_PATTERN[0] in output and IP_PATTERN[1] in output:
            host = output.split(IP_PATTERN[0])[1].split(IP_PATTERN[1])[0]
            logger.info(f"Found host: {host}")
            return host
    except (IndexError, AttributeError) as e:
        logger.debug(f"Could not extract host: {e}")
    
    return None


def monitor_bsub_job(job_id: str, timeout: int = 300) -> Optional[str]:
    """
    Monitor a bsub job and extract host information from its output.
    
    Args:
        job_id: LSF job ID to monitor
        timeout: Maximum time to wait in seconds
        
    Returns:
        Host URL if found, None if job finished without host info
    """
    command = f"bpeek {job_id}"
    attempts = 0
    max_attempts = timeout // 5  # Check every 5 seconds
    
    logger.info(f"Monitoring job {job_id} for host information...")
    
    while attempts < max_attempts:
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            output, error_output = process.communicate(timeout=5)
            
            # Check if job hasn't started yet
            if f"Job <{job_id}> : Not yet started." in error_output:
                logger.debug(f"Job {job_id} not yet started. Waiting...")
                attempts += 1
                continue
            
            # Check if job has finished
            if output == "" and process.poll() is not None:
                logger.warning(f"Job {job_id} has finished without host info")
                return None
            
            # Try to extract host
            if output:
                host = extract_host_from_output(output)
                if host:
                    return host
                
                # Check for errors
                if "error" in output.lower():
                    logger.error(f"Error in job output: {output}")
            
            attempts += 1
            
        except subprocess.TimeoutExpired:
            logger.debug(f"Timeout waiting for job {job_id} output")
            attempts += 1
        except Exception as e:
            logger.error(f"Error monitoring job {job_id}: {e}")
            return None
    
    logger.warning(f"Timeout waiting for host from job {job_id}")
    return None


def run_locally(command: str, name: str) -> Job:
    """
    Run command locally as a subprocess (fallback when bsub unavailable).
    
    Args:
        command: Shell command to execute
        name: Job name for tracking
        
    Returns:
        Job object with process and host information
    """
    logger.info(f"Running locally: {command}")
    
    process = subprocess.Popen(
        command.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output = ""
    max_wait = 60  # seconds
    waited = 0
    
    while waited < max_wait:
        # Non-blocking read with timeout
        rlist, _, _ = select.select(
            [process.stdout, process.stderr], [], [], 1.0
        )

        # Read available output
        if process.stdout in rlist:
            line = process.stdout.readline()
            if line:
                output += line
        
        if process.stderr in rlist:
            line = process.stderr.readline()
            if line:
                output += line
        
        # Try to extract host
        host = extract_host_from_output(output)
        if host:
            job = Job(model_name=name, host=host, process=process)
            g.jobs.append(job)
            return job

        # Check if process died
        if process.poll() is not None:
            logger.error(f"Process exited prematurely: {output}")
            break
        
        waited += 1
    
    logger.warning(f"Could not extract host from local process after {max_wait}s")
    job = Job(model_name=name, host=None, process=process)
    g.jobs.append(job)
    return job


def start_hosts(
    command: str,
    queue: str = DEFAULT_QUEUE,
    charge_group: Optional[str] = None,
    job_name: str = "example_job",
    use_https: bool = False,
) -> Job:
    """
    Start a server job either via bsub or locally.
    
    Args:
        command: Command to execute
        queue: LSF queue name (for bsub)
        charge_group: Project for billing (for bsub)
        job_name: Name for the job
        use_https: Whether to use HTTPS (adds cert/key flags)
        
    Returns:
        Job object with job information
    """
    # Update global settings
    g.queue = queue
    g.charge_group = charge_group
    
    # Add HTTPS flags if needed
    if use_https:
        command = f"{command} --certfile=host.cert --keyfile=host.key"
    
    if is_bsub_available():
        logger.info("Using bsub for job submission")
        try:
            result = submit_bsub_job(
                command,
                queue,
                charge_group,
                job_name=f"{job_name}_server"
            )
            
            # Extract job ID from output like "Job <12345> is submitted..."
            job_id = result.stdout.split()[1].strip('<>')
            host = monitor_bsub_job(job_id)
            
            job = Job(job_id=job_id, model_name=job_name, host=host)
            g.jobs.append(job)
            return job
            
        except Exception as e:
            logger.error(f"Failed to submit bsub job: {e}")
            logger.info("Falling back to local execution")
            return run_locally(command, job_name)
    else:
        logger.info("bsub not available, running locally")
        return run_locally(command, job_name)
