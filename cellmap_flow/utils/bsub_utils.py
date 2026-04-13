"""
Utilities for job submission and management across different execution environments.

Supports:
- LSF (bsub) cluster jobs
- Local process execution
- Extensible to cloud providers and other cluster types
"""

import subprocess
import logging
import sys
import signal
import select
import time
from typing import Optional, List
from abc import ABC, abstractmethod
from enum import Enum

from cellmap_flow.globals import g
from cellmap_flow.utils.web_utils import IP_PATTERN

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SECURITY = "http"
DEFAULT_QUEUE = "gpu_h100"
DEFAULT_CHARGE_GROUP = "cellmap"
SERVER_COMMAND = "cellmap_flow_server"


class JobStatus(Enum):
    """Enumeration of possible job statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class Job(ABC):
    """
    Abstract base class for jobs across different execution environments.
    
    Subclasses should implement:
    - kill(): Terminate the job
    - get_status(): Get current job status
    - wait_for_host(): Wait for and extract host information
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.status = JobStatus.RUNNING
        self.host: Optional[str] = None
    
    @abstractmethod
    def kill(self) -> None:
        """Terminate the job."""
        pass
    
    @abstractmethod
    def get_status(self) -> JobStatus:
        """Get the current status of the job."""
        pass
    
    @abstractmethod
    def wait_for_host(self, timeout: int = 300) -> Optional[str]:
        """
        Wait for the job to provide host information.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Host URL if found, None otherwise
        """
        pass
    
    def is_running(self) -> bool:
        """Check if the job is currently running."""
        return self.status == JobStatus.RUNNING


class LocalJob(Job):
    """Job running as a local subprocess."""
    
    def __init__(self, process: subprocess.Popen, model_name: Optional[str] = None):
        super().__init__(model_name)
        self.process = process
    
    def kill(self) -> None:
        """Terminate the local process."""
        if self.process is None or self.process.poll() is not None:
            logger.warning("Local job is not running.")
            return
        
        logger.info(f"Killing local process {self.process.pid}")
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process didn't terminate, killing forcefully")
                self.process.kill()
                self.process.wait()
        except Exception as e:
            logger.error(f"Error killing process: {e}")
        finally:
            self.status = JobStatus.KILLED
    
    def get_status(self) -> JobStatus:
        """Get current status by checking process state."""
        if self.process is None:
            return JobStatus.FAILED
        
        returncode = self.process.poll()
        if returncode is None:
            return JobStatus.RUNNING
        elif returncode == 0:
            return JobStatus.COMPLETED
        else:
            return JobStatus.FAILED
    
    def wait_for_host(self, timeout: int = 180) -> Optional[str]:
        """
        Monitor process output for host information.

        Args:
            timeout: Maximum time to wait in seconds (default 180s for model loading)

        Returns:
            Host URL if found, None otherwise
        """
        if self.host:
            return self.host
        
        logger.info(f"Monitoring local process for host information...")
        output = ""
        waited = 0
        
        while waited < timeout:
            # Non-blocking read with 1 second timeout
            rlist, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1.0
            )

            # Read available output
            if self.process.stdout in rlist:
                line = self.process.stdout.readline()
                if line:
                    output += line
            
            if self.process.stderr in rlist:
                line = self.process.stderr.readline()
                if line:
                    output += line
            
            # Try to extract host
            host = extract_host_from_output(output)
            if host:
                self.host = host
                logger.info(f"Found host: {host}")
                return host

            # Check if process died
            if self.process.poll() is not None:
                logger.error(f"Process exited prematurely with code {self.process.returncode}")
                self.status = JobStatus.FAILED
                break
            
            waited += 1
        
        logger.warning(f"Could not extract host from local process after {timeout}s")
        return None


class LSFJob(Job):
    """Job submitted to LSF cluster via bsub."""
    
    def __init__(self, job_id: str, model_name: Optional[str] = None):
        super().__init__(model_name)
        self.job_id = job_id
    
    def kill(self) -> None:
        """Terminate the LSF job using bkill."""
        logger.info(f"Killing LSF job {self.job_id}")
        try:
            result = subprocess.run(
                ["bkill", self.job_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.status = JobStatus.KILLED
                logger.info(f"Successfully killed job {self.job_id}")
            else:
                logger.error(f"Failed to kill job {self.job_id}: {result.stderr}")
        except Exception as e:
            logger.error(f"Error killing LSF job {self.job_id}: {e}")
    
    def get_status(self) -> JobStatus:
        """Query LSF for job status using bjobs."""
        try:
            result = subprocess.run(
                ["bjobs", "-noheader", self.job_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                # Job not found, likely completed or killed
                return self.status
            
            output = result.stdout.strip()
            if not output:
                return self.status
            
            # Parse bjobs output (format: JOBID USER STAT QUEUE FROM_HOST EXEC_HOST JOB_NAME SUBMIT_TIME)
            fields = output.split()
            if len(fields) >= 3:
                stat = fields[2]
                if stat == "RUN":
                    return JobStatus.RUNNING
                elif stat == "PEND":
                    return JobStatus.PENDING
                elif stat == "DONE":
                    return JobStatus.COMPLETED
                elif stat == "EXIT":
                    return JobStatus.FAILED
            
            return self.status
        except Exception as e:
            logger.debug(f"Error checking LSF job status: {e}")
            return self.status
    
    def wait_for_host(self, timeout: int = 300) -> Optional[str]:
        """
        Monitor LSF job output using bpeek to extract host information.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Host URL if found, None otherwise
        """
        if self.host:
            return self.host
        
        logger.info(f"Monitoring LSF job {self.job_id} for host information...")
        
        attempts = 0
        max_attempts = timeout * 2  # Check every 0.5 seconds
        pending_time = 0
        warned_pending_30s = False
        warned_pending_60s = False
        
        while attempts < max_attempts:
            try:
                # Check job status first
                current_status = self.get_status()
                
                # Track pending time and warn if too long
                if current_status == JobStatus.PENDING:
                    pending_time += 0.5
                    
                    if pending_time >= 30 and not warned_pending_30s:
                        logger.warning(f"Job {self.job_id} has been pending for {pending_time}s. "
                                     f"Queue may be busy or resources unavailable.")
                        warned_pending_30s = True
                    
                    if pending_time >= 60 and not warned_pending_60s:
                        logger.warning(f"Job {self.job_id} still pending after {pending_time}s. "
                                     f"Consider checking queue status or resource availability.")
                        warned_pending_60s = True
                    
                    if pending_time >= 120:
                        logger.warning(f"Job {self.job_id} pending for {pending_time}s. "
                                     f"This is unusually long. You may want to check with 'bjobs {self.job_id}'")
                else:
                    # Reset pending time when job starts running
                    if pending_time > 0:
                        logger.info(f"Job {self.job_id} started after {pending_time}s in pending state")
                        pending_time = 0
                
                result = subprocess.run(
                    ["bpeek", self.job_id],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                output = result.stdout
                error = result.stderr
                
                # Check if job hasn't started yet
                if f"Job <{self.job_id}> : Not yet started." in error:
                    logger.debug(f"Job {self.job_id} not yet started. Waiting...")
                    attempts += 1
                    time.sleep(0.5)  # Wait 0.5 seconds before next check
                    continue
                
                # Check if job has finished
                if not output and result.returncode != 0:
                    logger.warning(f"Job {self.job_id} may have finished")
                    break
                
                # Try to extract host
                if output:
                    host = extract_host_from_output(output)
                    if host:
                        self.host = host
                        logger.info(f"Found host: {host}")
                        return host
                    
                    # Check for errors
                    if "error" in output.lower():
                        logger.error(f"Error in job output: {output}")
                
                attempts += 1
                time.sleep(0.5)  # Wait 0.5 seconds before next check
                
            except subprocess.TimeoutExpired:
                logger.debug(f"Timeout waiting for job {self.job_id} output")
                attempts += 1
                time.sleep(0.5)  # Wait 0.5 seconds before next check
            except Exception as e:
                logger.error(f"Error monitoring job {self.job_id}: {e}")
                break
        
        logger.warning(f"Timeout waiting for host from job {self.job_id}")
        return None


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
            return host
    except (IndexError, AttributeError) as e:
        logger.debug(f"Could not extract host: {e}")
    
    return None


def cleanup_handler(signum: int, frame) -> None:
    """
    Signal handler for graceful shutdown.
    Kills all tracked jobs before exiting.
    """
    logger.warning(f"Received signal {signum}. Cleaning up jobs...")
    for job in g.jobs:
        logger.info(f"Killing job: {job.model_name}")
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
) -> LSFJob:
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
        LSFJob object for the submitted job
        
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
        
        # Extract job ID from output like "Job <12345> is submitted..."
        job_id = result.stdout.split()[1].strip('<>')
        logger.info(f"Job {job_id} submitted successfully")
        
        return LSFJob(job_id=job_id, model_name=job_name)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Job submission failed: {e.stderr}")
        if not charge_group:
            logger.error("Hint: You may need to specify a charge group with -P option")
        raise
    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        raise


def run_locally(command: str, name: str) -> LocalJob:
    """
    Run command locally as a subprocess (fallback when bsub unavailable).
    
    Args:
        command: Shell command to execute
        name: Job name for tracking
        
    Returns:
        LocalJob object with process information
    """
    logger.info(f"Running locally: {command}")

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        local_job = LocalJob(process=process, model_name=name)
        return local_job

    except Exception as e:
        logger.error(f"Error starting local process: {e}")
        raise


def start_hosts(
    command: str,
    queue: str = DEFAULT_QUEUE,
    charge_group: Optional[str] = None,
    job_name: str = "example_job",
    use_https: bool = False,
    wait_for_host: bool = True,
) -> Job:
    """
    Start a server job either via bsub or locally.
    
    Args:
        command: Command to execute
        queue: LSF queue name (for bsub)
        charge_group: Project for billing (for bsub)
        job_name: Name for the job
        use_https: Whether to use HTTPS (adds cert/key flags)
        wait_for_host: Whether to wait for host information before returning
        
    Returns:
        Job object (LSFJob or LocalJob) with job information
    """
    # Update global settings
    g.queue = queue
    g.charge_group = charge_group
    
    # Add HTTPS flags if needed
    if use_https:
        command = f"{command} --certfile=host.cert --keyfile=host.key"
    
    job: Job
    
    if is_bsub_available():
        logger.info("Using bsub for job submission")
        try:
            job = submit_bsub_job(
                command,
                queue,
                charge_group,
                job_name=f"{job_name}"
            )
            
            if wait_for_host:
                job.wait_for_host()
            
            g.jobs.append(job)
            return job
            
        except Exception as e:
            logger.error(f"Failed to submit bsub job: {e}")
            logger.info("Falling back to local execution")
    else:
        logger.info("bsub not available, running locally")
    
    # Local execution (either by choice or as fallback)
    job = run_locally(command, job_name)
    
    if wait_for_host:
        job.wait_for_host()
    
    g.jobs.append(job)
    return job
