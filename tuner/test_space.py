import json
import os
import unittest

from .space import Space


def param(name):
    return {"name": name, "default": 0, "low": -1, "high": 1}


class IndexedParamTests(unittest.TestCase):
    def test_nested_and_negative_indices(self):
        space = Space([param("a[-1][2]"), param("a[0][0]"), param("b[1]"), param("x")])
        base = {"a": [[0, 0, 0], [0, 0, 0]], "b": [0, 0]}
        out = space.config({"a[-1][2]": 0.5, "a[0][0]": 0.75, "b[1]": 0.25, "x": 0.1}, base)
        self.assertEqual(out, {"a": [[0.75, 0, 0], [0, 0, 0.5]], "b": [0, 0.25], "x": 0.1})
        self.assertEqual(base["a"], [[0, 0, 0], [0, 0, 0]])

    def test_last_row_follows_appended_rows(self):
        space = Space([param("a[-1][0]")])
        out = space.config({"a[-1][0]": 0.5}, {"a": [[1, 1], [1, 1], [0, 0]]})
        self.assertEqual(out["a"], [[1, 1], [1, 1], [0.5, 0]])

    def test_shape_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            Space([param("a[-1][2]")]).config({"a[-1][2]": 0.5}, {"a": [0, 0, 0]})
        with self.assertRaises(ValueError):
            Space([param("a[0][5]")]).config({"a[0][5]": 0.5}, {"a": [[0, 0]]})
        with self.assertRaises(ValueError):
            Space([param("a[1]")]).config({"a[1]": 0.5}, {"a": [[0], [0]]})

    def test_conditional_space_tunes_the_last_row(self):
        path = os.path.join(os.path.dirname(__file__), "spaces", "conditional.json")
        with open(path) as f:
            space = Space(json.load(f))
        base = {"opp_cond_u": [[1.0] * 19, [0.0] * 19], "opp_cond_mean": [[1.0] * 19, [0.0] * 19],
                "opp_cond_v": [[1.0] * 48, [0.0] * 48], "opp_cond_reward": [[1.0, 1.0], [0.0, 0.0]]}
        out = space.config({p.name: p.high for p in space.params}, base)
        for key in base:
            self.assertEqual(out[key][0], base[key][0])
            self.assertNotEqual(out[key][1], base[key][1])


if __name__ == "__main__":
    unittest.main()
