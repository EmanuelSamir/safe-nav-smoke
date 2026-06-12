from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseRenderer(ABC):
    """Abstract base class defining the contract for all rendering classes."""

    @abstractmethod
    def render(self, info: Dict[str, Any]) -> Any:
        """Render a single frame based on the state/environment info."""
        pass

    @abstractmethod
    def save_frame(self) -> None:
        """Save the current frame to buffer/list."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up and close rendering resources."""
        pass
