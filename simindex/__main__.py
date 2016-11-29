"""Allow simindex to be executable through `python -m simindex`."""
from __future__ import absolute_import
from .cli import main


if __name__ == "__main__":  # pragma: no cover
        main(prog_name="simindex")
