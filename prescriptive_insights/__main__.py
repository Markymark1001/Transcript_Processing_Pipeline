"""
CLI entry point for prescriptive_insights package.

Allows running: python -m prescriptive_insights.chunk_builder --rebuild
"""

from .chunk_builder import main

if __name__ == "__main__":
    main()