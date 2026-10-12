from pathlib import Path
# Set package search path to repository root so trustagent.<subpackage> resolves directly
__path__ = [str(Path(__file__).resolve().parent.parent)]
