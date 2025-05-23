"""
Cost tracking module for experiment processors
"""

import threading
from typing import Optional


class CostTracker:
    """
    Singleton class for tracking costs across threads
    
    This class provides thread-safe cost tracking capabilities for experiments.
    It's implemented as a singleton to ensure the same cost tracker is used
    across different threads or processes.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CostTracker, cls).__new__(cls)
                cls._instance.total_cost = 0.0
                cls._instance.cost_threshold = None
                cls._instance.should_stop = False
        return cls._instance
    
    def reset(self, cost_threshold: Optional[float] = None) -> None:
        """
        Reset cost tracker state
        
        Args:
            cost_threshold: Maximum cost allowed before stopping (in USD)
        """
        with self._lock:
            self.total_cost = 0.0
            self.cost_threshold = cost_threshold
            self.should_stop = False
    
    def update_cost(self, cost: float) -> float:
        """
        Update total cost and check threshold
        
        Args:
            cost: Cost to add (in USD)
            
        Returns:
            Updated total cost
        """
        with self._lock:
            self.total_cost += cost
            if self.cost_threshold and self.total_cost >= self.cost_threshold:
                self.should_stop = True
            return self.total_cost
    
    @property
    def stop_requested(self) -> bool:
        """
        Check if execution should stop due to cost threshold
        
        Returns:
            True if cost threshold has been reached, False otherwise
        """
        with self._lock:
            return self.should_stop 