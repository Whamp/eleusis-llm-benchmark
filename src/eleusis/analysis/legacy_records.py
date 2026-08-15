"""Type aliases for heterogeneous historical analysis result files."""

from typing import Any, TypeAlias

# Analysis must read result files written before the current validated schemas.
# Their nested config, token, and turn shapes differ by benchmark version, so Any
# is intentionally contained at this decode boundary instead of leaking into game code.
LegacyRecord: TypeAlias = dict[str, Any]
LegacyResults: TypeAlias = list[LegacyRecord]
RuleLookup: TypeAlias = dict[str, LegacyRecord]
