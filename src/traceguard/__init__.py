"""TRACEGUARD: empirical research framework for reasoning-aware monitors.

TRACEGUARD studies whether access to an agent's generated reasoning trace
improves a monitor's ability to identify unauthorized agent behavior beyond
observable tool actions, under a controlled, deterministic simulation.

This package implements the research protocol described in
``docs/01_RESEARCH_PROTOCOL.md`` and the architecture in
``docs/02_ARCHITECTURE.md``. Nothing in this package fabricates evidence:
every episode, score, and figure is derived from recorded, hashed artifacts.
"""

__version__ = "0.2.0"

SCHEMA_VERSION = "1.0.0"
