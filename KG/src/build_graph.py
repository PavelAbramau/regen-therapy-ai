"""Back-compat entrypoint: core graph assembly lives in `build_core_graph`."""

from __future__ import annotations

from build_core_graph import main


if __name__ == "__main__":
    main()
