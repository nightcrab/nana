"""The parameters being tuned, and the map between them and the optimiser's unit cube.

A space is a JSON list of entries:
    {"name": "height", "default": -46, "low": -150, "high": 0}
    {"name": "clears[1]", "default": -1700, "low": -3400, "high": 0}   # one element of an array key
    {"name": "c_base", "default": 19652, "low": 2000, "high": 200000, "log": true}
    {"name": "depth_step", "default": 3, "low": 1, "high": 8, "integer": true}

Names are NANA_CONFIG keys (see cpp/Tetris/bot/Config.hpp), with [i] for array elements. The
optimiser works in [0, 1]^d; `log` entries are uniform in log space (low and high must be positive).
"""

import json
import math
import re

_INDEXED = re.compile(r"^(\w+)\[(\d+)\]$")


class Param:
    def __init__(self, spec):
        self.name = spec["name"]
        self.default = float(spec["default"])
        self.low = float(spec["low"])
        self.high = float(spec["high"])
        self.log = bool(spec.get("log", False))
        self.integer = bool(spec.get("integer", False))
        if not self.low < self.high:
            raise ValueError(f"{self.name}: low must be below high")
        if not self.low <= self.default <= self.high:
            raise ValueError(f"{self.name}: default {self.default} outside [{self.low}, {self.high}]")
        if self.log and self.low <= 0:
            raise ValueError(f"{self.name}: log scale needs positive bounds")
        m = _INDEXED.match(self.name)
        self.key, self.index = (m.group(1), int(m.group(2))) if m else (self.name, None)

    def to_unit(self, value):
        if self.log:
            return (math.log(value) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (value - self.low) / (self.high - self.low)

    def from_unit(self, u):
        u = min(max(u, 0.0), 1.0)
        if self.log:
            value = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            value = self.low + u * (self.high - self.low)
        return round(value) if self.integer else value

    def spec(self):
        out = {"name": self.name, "default": self.default, "low": self.low, "high": self.high}
        if self.log:
            out["log"] = True
        if self.integer:
            out["integer"] = True
        return out


class Space:
    def __init__(self, specs):
        self.params = [Param(s) for s in specs]
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError("a parameter is listed twice")

    @staticmethod
    def load(path):
        with open(path) as f:
            return Space(json.load(f))

    def __len__(self):
        return len(self.params)

    def names(self):
        return [p.name for p in self.params]

    def default_unit(self):
        return [p.to_unit(p.default) for p in self.params]

    def values(self, unit):
        """{name: value} for a point in the unit cube."""
        return {p.name: p.from_unit(u) for p, u in zip(self.params, unit)}

    def unit(self, values):
        """A point in the unit cube for {name: value}; missing names take their defaults."""
        return [p.to_unit(values.get(p.name, p.default)) for p in self.params]

    def config(self, values, base=None):
        """A NANA_CONFIG dict: `base` (a full config, e.g. from nana_play --print-config) with these
        values written over it. Array elements are written into a copy of the base array."""
        out = json.loads(json.dumps(base)) if base else {}
        for p in self.params:
            if p.name not in values:
                continue
            if p.index is None:
                out[p.key] = values[p.name]
            else:
                if p.key not in out:
                    raise ValueError(f"{p.name}: the base config has no array {p.key}")
                out[p.key][p.index] = values[p.name]
        return out

    def extend(self, other):
        """A space with other's parameters added (for --add-params); existing names are kept."""
        have = set(self.names())
        return Space([p.spec() for p in self.params] + [p.spec() for p in other.params if p.name not in have])

    def specs(self):
        return [p.spec() for p in self.params]
