"""
Example extensions of the Job class for cloud providers.

This demonstrates how to extend the abstract Job class to support
different execution environments like AWS Batch, Google Cloud, Azure Batch, etc.
"""

from typing import Optional
import subprocess
import logging

from cellmap_flow.utils.bsub_utils import Job, JobStatus, extract_host_from_output

logger = logging.getLogger(__name__)


class AWSBatchJob(Job):
    """Job submitted to AWS Batch."""
    
    def __init__(self, job_id: str, job_queue: str, model_name: Optional[str] = None):
        super().__init__(model_name)
        self.job_id = job_id
        self.job_queue = job_queue
    
    def kill(self) -> None:
        """Terminate the AWS Batch job."""
        logger.info(f"Terminating AWS Batch job {self.job_id}")
        try:
            # Using AWS CLI
            result = subprocess.run(
                ["aws", "batch", "terminate-job", 
                 "--job-id", self.job_id,
                 "--reason", "Terminated by user"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                self.status = JobStatus.KILLED
                logger.info(f"Successfully terminated job {self.job_id}")
            else:
                logger.error(f"Failed to terminate job: {result.stderr}")
        except Exception as e:
            logger.error(f"Error terminating AWS Batch job: {e}")
    
    def get_status(self) -> JobStatus:
        """Query AWS Batch for job status."""
        try:
            result = subprocess.run(
                ["aws", "batch", "describe-jobs", "--jobs", self.job_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return self.status
            
            import json
            data = json.loads(result.stdout)
            
            if data.get('jobs'):
                job_status = data['jobs'][0].get('status')
                status_map = {
                    'SUBMITTED': JobStatus.PENDING,
                    'PENDING': JobStatus.PENDING,
                    'RUNNABLE': JobStatus.PENDING,
                    'STARTING': JobStatus.PENDING,
                    'RUNNING': JobStatus.RUNNING,
                    'SUCCEEDED': JobStatus.COMPLETED,
                    'FAILED': JobStatus.FAILED,
                }
                return status_map.get(job_status, self.status)
            
            return self.status
        except Exception as e:
            logger.debug(f"Error checking AWS Batch job status: {e}")
            return self.status
    
    def wait_for_host(self, timeout: int = 300) -> Optional[str]:
        """
        Monitor AWS Batch job logs for host information.
        
        This would typically involve:
        1. Getting the CloudWatch log stream for the job
        2. Monitoring the logs for host information
        3. Extracting the host URL
        """
        logger.info(f"Monitoring AWS Batch job {self.job_id} for host information...")
        
        # Implementation would query CloudWatch logs
        # This is a simplified example
        try:
            import time
            waited = 0
            
            while waited < timeout:
                # Get log events from CloudWatch
                # result = subprocess.run([
                #     "aws", "logs", "get-log-events",
                #     "--log-group-name", f"/aws/batch/job",
                #     "--log-stream-name", self._get_log_stream_name()
                # ], capture_output=True, text=True, timeout=10)
                
                # For demo purposes, just check job status
                status = self.get_status()
                if status == JobStatus.RUNNING:
                    # In real implementation, parse CloudWatch logs here
                    logger.debug(f"Job running, waiting for logs...")
                elif status != JobStatus.PENDING:
                    logger.warning(f"Job in unexpected state: {status}")
                    break
                
                time.sleep(5)
                waited += 5
            
            return None
        except Exception as e:
            logger.error(f"Error monitoring AWS Batch job: {e}")
            return None


class GoogleCloudJob(Job):
    """Job submitted to Google Cloud (Compute Engine or Batch)."""
    
    def __init__(self, job_id: str, project_id: str, region: str, model_name: Optional[str] = None):
        super().__init__(model_name)
        self.job_id = job_id
        self.project_id = project_id
        self.region = region
    
    def kill(self) -> None:
        """Terminate the Google Cloud job."""
        logger.info(f"Terminating Google Cloud job {self.job_id}")
        try:
            result = subprocess.run(
                ["gcloud", "batch", "jobs", "delete", self.job_id,
                 f"--project={self.project_id}",
                 f"--location={self.region}",
                 "--quiet"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                self.status = JobStatus.KILLED
                logger.info(f"Successfully deleted job {self.job_id}")
            else:
                logger.error(f"Failed to delete job: {result.stderr}")
        except Exception as e:
            logger.error(f"Error terminating Google Cloud job: {e}")
    
    def get_status(self) -> JobStatus:
        """Query Google Cloud for job status."""
        try:
            result = subprocess.run(
                ["gcloud", "batch", "jobs", "describe", self.job_id,
                 f"--project={self.project_id}",
                 f"--location={self.region}",
                 "--format=json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return self.status
            
            import json
            data = json.loads(result.stdout)
            
            job_status = data.get('status', {}).get('state')
            status_map = {
                'STATE_UNSPECIFIED': JobStatus.PENDING,
                'QUEUED': JobStatus.PENDING,
                'SCHEDULED': JobStatus.PENDING,
                'RUNNING': JobStatus.RUNNING,
                'SUCCEEDED': JobStatus.COMPLETED,
                'FAILED': JobStatus.FAILED,
                'DELETION_IN_PROGRESS': JobStatus.KILLED,
            }
            return status_map.get(job_status, self.status)
        except Exception as e:
            logger.debug(f"Error checking Google Cloud job status: {e}")
            return self.status
    
    def wait_for_host(self, timeout: int = 300) -> Optional[str]:
        """Monitor Google Cloud job logs for host information."""
        logger.info(f"Monitoring Google Cloud job {self.job_id} for host information...")
        
        # Implementation would query Cloud Logging
        # Similar pattern to AWS Batch
        return None


class AzureBatchJob(Job):
    """Job submitted to Azure Batch."""
    
    def __init__(self, job_id: str, batch_account: str, pool_id: str, model_name: Optional[str] = None):
        super().__init__(model_name)
        self.job_id = job_id
        self.batch_account = batch_account
        self.pool_id = pool_id
    
    def kill(self) -> None:
        """Terminate the Azure Batch job."""
        logger.info(f"Terminating Azure Batch job {self.job_id}")
        try:
            result = subprocess.run(
                ["az", "batch", "job", "delete",
                 "--job-id", self.job_id,
                 "--account-name", self.batch_account,
                 "--yes"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                self.status = JobStatus.KILLED
                logger.info(f"Successfully deleted job {self.job_id}")
            else:
                logger.error(f"Failed to delete job: {result.stderr}")
        except Exception as e:
            logger.error(f"Error terminating Azure Batch job: {e}")
    
    def get_status(self) -> JobStatus:
        """Query Azure Batch for job status."""
        try:
            result = subprocess.run(
                ["az", "batch", "job", "show",
                 "--job-id", self.job_id,
                 "--account-name", self.batch_account,
                 "--query", "state",
                 "--output", "tsv"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return self.status
            
            job_status = result.stdout.strip()
            status_map = {
                'active': JobStatus.RUNNING,
                'disabling': JobStatus.RUNNING,
                'disabled': JobStatus.FAILED,
                'enabling': JobStatus.PENDING,
                'terminating': JobStatus.KILLED,
                'completed': JobStatus.COMPLETED,
                'deleting': JobStatus.KILLED,
            }
            return status_map.get(job_status, self.status)
        except Exception as e:
            logger.debug(f"Error checking Azure Batch job status: {e}")
            return self.status
    
    def wait_for_host(self, timeout: int = 300) -> Optional[str]:
        """Monitor Azure Batch job for host information."""
        logger.info(f"Monitoring Azure Batch job {self.job_id} for host information...")
        # Implementation would query Azure Monitor or task output
        return None


class SlurmJob(Job):
    """Job submitted to SLURM cluster (common in HPC environments)."""
    
    def __init__(self, job_id: str, model_name: Optional[str] = None):
        super().__init__(model_name)
        self.job_id = job_id
    
    def kill(self) -> None:
        """Cancel the SLURM job using scancel."""
        logger.info(f"Cancelling SLURM job {self.job_id}")
        try:
            result = subprocess.run(
                ["scancel", self.job_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.status = JobStatus.KILLED
                logger.info(f"Successfully cancelled job {self.job_id}")
            else:
                logger.error(f"Failed to cancel job: {result.stderr}")
        except Exception as e:
            logger.error(f"Error cancelling SLURM job: {e}")
    
    def get_status(self) -> JobStatus:
        """Query SLURM for job status using squeue/sacct."""
        try:
            # Try squeue first (for running/pending jobs)
            result = subprocess.run(
                ["squeue", "-j", self.job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                job_status = result.stdout.strip()
                status_map = {
                    'PENDING': JobStatus.PENDING,
                    'RUNNING': JobStatus.RUNNING,
                    'SUSPENDED': JobStatus.RUNNING,
                    'COMPLETING': JobStatus.RUNNING,
                    'COMPLETED': JobStatus.COMPLETED,
                    'FAILED': JobStatus.FAILED,
                    'TIMEOUT': JobStatus.FAILED,
                    'CANCELLED': JobStatus.KILLED,
                    'NODE_FAIL': JobStatus.FAILED,
                }
                return status_map.get(job_status, self.status)
            
            # Job not in queue, check sacct for completed jobs
            result = subprocess.run(
                ["sacct", "-j", self.job_id, "-n", "-o", "State"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                job_status = result.stdout.strip().split()[0]  # Get first status
                if 'COMPLETED' in job_status:
                    return JobStatus.COMPLETED
                elif 'FAILED' in job_status:
                    return JobStatus.FAILED
                elif 'CANCELLED' in job_status:
                    return JobStatus.KILLED
            
            return self.status
        except Exception as e:
            logger.debug(f"Error checking SLURM job status: {e}")
            return self.status
    
    def wait_for_host(self, timeout: int = 300) -> Optional[str]:
        """Monitor SLURM job output file for host information."""
        logger.info(f"Monitoring SLURM job {self.job_id} for host information...")
        
        import time
        waited = 0
        output_file = f"slurm-{self.job_id}.out"
        
        while waited < timeout:
            try:
                # Try to read the output file
                with open(output_file, 'r') as f:
                    output = f.read()
                    host = extract_host_from_output(output)
                    if host:
                        self.host = host
                        logger.info(f"Found host: {host}")
                        return host
            except FileNotFoundError:
                pass  # Output file not created yet
            except Exception as e:
                logger.debug(f"Error reading output file: {e}")
            
            # Check if job is still running
            status = self.get_status()
            if status not in [JobStatus.PENDING, JobStatus.RUNNING]:
                logger.warning(f"Job in non-running state: {status}")
                break
            
            time.sleep(5)
            waited += 5
        
        logger.warning(f"Could not extract host from SLURM job after {timeout}s")
        return None


# Example usage and factory function
def create_job_for_environment(env_type: str, **kwargs) -> Job:
    """
    Factory function to create appropriate Job subclass based on environment.
    
    Args:
        env_type: Type of environment ('lsf', 'aws', 'gcp', 'azure', 'slurm', 'local')
        **kwargs: Environment-specific parameters
        
    Returns:
        Appropriate Job subclass instance
        
    Example:
        >>> job = create_job_for_environment('aws', job_id='12345', job_queue='my-queue')
        >>> job = create_job_for_environment('lsf', job_id='67890')
    """
    from cellmap_flow.utils.bsub_utils import LSFJob, LocalJob
    
    env_map = {
        'lsf': LSFJob,
        'aws': AWSBatchJob,
        'gcp': GoogleCloudJob,
        'azure': AzureBatchJob,
        'slurm': SlurmJob,
        # 'local' is handled separately as it needs a process
    }
    
    if env_type == 'local':
        raise ValueError("Use run_locally() to create LocalJob instances")
    
    job_class = env_map.get(env_type)
    if not job_class:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    return job_class(**kwargs)
