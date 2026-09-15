class AppError(Exception):
    """Base application error."""


class ModelLoadError(AppError):
    """Raised when model weights/config cannot be loaded."""


class PipelineError(AppError):
    """Raised by pipeline workers when a stage fails."""
