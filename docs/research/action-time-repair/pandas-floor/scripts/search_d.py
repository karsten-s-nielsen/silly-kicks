"""Form-D search: a comparison on an array taken with a dtype-less `.to_numpy()` / `.values`, whose result
feeds a boolean `&` / `|` / `~` -- the pandas-2.1 failure is `pd.NA` objects from a nullable integer column.

For every function: names bound from `<subject>.to_numpy()` with no `dtype=` (or `<subject>.values`, or a
slice of such a name, e.g. `prev[1:] = x[:-1]` binds nothing new but `np.empty(..., dtype=x.dtype)` is
followed via the `prev_` copy pattern). A Compare that touches such a name and sits inside a BitAnd/BitOr/
Invert is a hit. Prints file:line, the bound names in the compare, and the subject column literal(s).

Usage, from the repository root: ``python docs/research/action-time-repair/pandas-floor/scripts/search_d.py .``
"""
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])


def _subject_literal(node: ast.AST) -> str:
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return str(node.slice.value)
    if isinstance(node, ast.Attribute):
        return "." + node.attr
    return "<expr>"


def _dtypeless_array(value: ast.AST):
    """`<subj>.to_numpy()` without dtype, or `<subj>.values` -> subject literal; else None."""
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "to_numpy":
        if not any(k.arg == "dtype" for k in value.keywords) and not value.args:
            return _subject_literal(value.func.value)
    if isinstance(value, ast.Attribute) and value.attr == "values":
        return _subject_literal(value.value)
    return None


hits = []
for path in sorted(list(root.joinpath("silly_kicks").rglob("*.py")) + list(root.joinpath("scripts").rglob("*.py"))):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        continue
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bound: dict[str, str] = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                lit = _dtypeless_array(node.value)
                if lit is not None:
                    bound[node.targets[0].id] = lit
        # the prev_* copy pattern: `prev = np.empty(n, dtype=x.dtype)` inherits x's object-ness
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                for kw in getattr(node.value, "keywords", []):
                    if (kw.arg == "dtype" and isinstance(kw.value, ast.Attribute) and kw.value.attr == "dtype"
                            and isinstance(kw.value.value, ast.Name) and kw.value.value.id in bound):
                        bound[node.targets[0].id] = bound[kw.value.value.id] + " (via dtype copy)"
        if not bound:
            continue
        for node in ast.walk(fn):
            ops = []
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
                ops = [node.left, node.right]
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
                ops = [node.operand]
            for operand in ops:
                for sub in ast.walk(operand):
                    if isinstance(sub, ast.Compare):
                        names = {n.id for n in ast.walk(sub) if isinstance(n, ast.Name) and n.id in bound}
                        if names:
                            hits.append((f"{path.relative_to(root).as_posix()}:{sub.lineno}",
                                         sorted(names), sorted({bound[n] for n in names})))
seen = set()
for where, names, subj in hits:
    if where in seen:
        continue
    seen.add(where)
    print(where, names, subj)
print("sites:", len(seen))
