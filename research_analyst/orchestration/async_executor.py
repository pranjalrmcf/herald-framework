"""
Async executor for the research analyst system.
Handles parallel execution of retrieval and processing tasks.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import List, Callable, Any, Optional, Dict
from functools import wraps
import time

from research_analyst.core.models import Document
from research_analyst.core.exceptions import TimeoutError as CustomTimeoutError
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger


logger = get_logger()


class AsyncExecutor:
    """Execute tasks in parallel with timeout and error handling."""
    
    def __init__(self):
        """Initialize async executor."""
        self.settings = get_settings()
        self.logger = get_logger()
        
        # Create thread pool
        self.executor = ThreadPoolExecutor(
            max_workers=self.settings.max_workers
        )
    
    def execute_parallel(
        self,
        tasks: List[Callable],
        timeout: Optional[int] = None
    ) -> List[Any]:
        """
        Execute multiple tasks in parallel.
        
        Args:
            tasks: List of callable tasks
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            List of results (None for failed tasks)
        """
        if timeout is None:
            timeout = self.settings.request_timeout
        
        self.logger.info(
            "Executing parallel tasks",
            num_tasks=len(tasks),
            timeout=timeout
        )
        
        start_time = time.time()
        
        # Submit all tasks
        futures = [self.executor.submit(task) for task in tasks]
        
        # Collect results
        results = []
        errors = []
        
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except FutureTimeoutError:
                self.logger.warning(
                    "Task timeout",
                    task_index=i,
                    timeout=timeout
                )
                results.append(None)
                errors.append(f"Task {i} timed out")
            except Exception as e:
                self.logger.warning(
                    "Task failed",
                    task_index=i,
                    error=str(e)
                )
                results.append(None)
                errors.append(f"Task {i} failed: {str(e)}")
        
        elapsed = (time.time() - start_time) * 1000  # ms
        
        self.logger.info(
            "Parallel execution complete",
            num_tasks=len(tasks),
            num_successful=sum(1 for r in results if r is not None),
            num_failed=len(errors),
            elapsed_ms=elapsed
        )
        
        return results
    
    def execute_with_retry(
        self,
        task: Callable,
        max_retries: Optional[int] = None,
        backoff_factor: float = 2.0
    ) -> Any:
        """
        Execute task with retry logic.
        
        Args:
            task: Task to execute
            max_retries: Maximum retries (uses default if None)
            backoff_factor: Backoff multiplier between retries
            
        Returns:
            Task result
            
        Raises:
            Exception: If all retries fail
        """
        if max_retries is None:
            max_retries = self.settings.max_retries
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                result = task()
                
                if attempt > 0:
                    self.logger.info(
                        "Task succeeded after retry",
                        attempt=attempt
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    wait_time = backoff_factor ** attempt
                    self.logger.warning(
                        "Task failed, retrying",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        wait_time=wait_time,
                        error=str(e)
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        "Task failed after all retries",
                        attempts=attempt + 1,
                        error=str(e)
                    )
        
        raise last_exception
    
    def batch_execute(
        self,
        items: List[Any],
        batch_processor: Callable[[List[Any]], List[Any]],
        batch_size: int = 10
    ) -> List[Any]:
        """
        Execute batch processing in parallel.
        
        Args:
            items: Items to process
            batch_processor: Function to process a batch
            batch_size: Size of each batch
            
        Returns:
            Combined results
        """
        self.logger.info(
            "Starting batch execution",
            num_items=len(items),
            batch_size=batch_size
        )
        
        # Split into batches
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]
        
        # Create tasks
        tasks = [
            lambda batch=batch: batch_processor(batch)
            for batch in batches
        ]
        
        # Execute in parallel
        batch_results = self.execute_parallel(tasks)
        
        # Flatten results
        all_results = []
        for batch_result in batch_results:
            if batch_result:
                all_results.extend(batch_result)
        
        self.logger.info(
            "Batch execution complete",
            num_batches=len(batches),
            num_results=len(all_results)
        )
        
        return all_results
    
    async def async_execute_parallel(
        self,
        async_tasks: List[Callable],
        timeout: Optional[int] = None
    ) -> List[Any]:
        """
        Execute async tasks in parallel.
        
        Args:
            async_tasks: List of async callable tasks
            timeout: Timeout in seconds
            
        Returns:
            List of results
        """
        if timeout is None:
            timeout = self.settings.request_timeout
        
        self.logger.info(
            "Executing async parallel tasks",
            num_tasks=len(async_tasks)
        )
        
        try:
            # Execute with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*[task() for task in async_tasks], return_exceptions=True),
                timeout=timeout
            )
            
            # Log exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(
                        "Async task failed",
                        task_index=i,
                        error=str(result)
                    )
            
            return results
            
        except asyncio.TimeoutError:
            self.logger.error(
                "Async execution timeout",
                timeout=timeout
            )
            raise CustomTimeoutError(
                "Async execution timeout",
                timeout_seconds=timeout
            )
    
    def execute_with_fallback(
        self,
        primary_task: Callable,
        fallback_task: Callable
    ) -> Any:
        """
        Execute task with fallback.
        
        Args:
            primary_task: Primary task to try
            fallback_task: Fallback if primary fails
            
        Returns:
            Result from primary or fallback
        """
        try:
            return primary_task()
        except Exception as e:
            self.logger.warning(
                "Primary task failed, using fallback",
                error=str(e)
            )
            try:
                return fallback_task()
            except Exception as e2:
                self.logger.error(
                    "Fallback task also failed",
                    error=str(e2)
                )
                raise
    
    def timed_execute(
        self,
        task: Callable,
        task_name: str = "task"
    ) -> tuple[Any, float]:
        """
        Execute task and measure time.
        
        Args:
            task: Task to execute
            task_name: Name for logging
            
        Returns:
            Tuple of (result, elapsed_ms)
        """
        start_time = time.time()
        
        try:
            result = task()
            elapsed_ms = (time.time() - start_time) * 1000
            
            self.logger.debug(
                f"{task_name} completed",
                elapsed_ms=elapsed_ms
            )
            
            return result, elapsed_ms
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.error(
                f"{task_name} failed",
                elapsed_ms=elapsed_ms,
                error=str(e)
            )
            raise
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown executor.
        
        Args:
            wait: Whether to wait for pending tasks
        """
        self.logger.info("Shutting down executor")
        self.executor.shutdown(wait=wait)
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Decorator for making functions retryable
def retryable(max_retries: int = 3, backoff_factor: float = 2.0):
    """
    Decorator to make functions retryable.
    
    Args:
        max_retries: Maximum number of retries
        backoff_factor: Backoff multiplier
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            executor = AsyncExecutor()
            task = lambda: func(*args, **kwargs)
            return executor.execute_with_retry(
                task,
                max_retries=max_retries,
                backoff_factor=backoff_factor
            )
        return wrapper
    return decorator


# Decorator for timing functions
def timed(task_name: Optional[str] = None):
    """
    Decorator to time function execution.
    
    Args:
        task_name: Optional name for logging
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = task_name or func.__name__
            executor = AsyncExecutor()
            task = lambda: func(*args, **kwargs)
            result, elapsed = executor.timed_execute(task, name)
            return result
        return wrapper
    return decorator