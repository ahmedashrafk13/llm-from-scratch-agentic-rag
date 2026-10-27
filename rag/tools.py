"""Tools available to the agent. All are offline ("web-free")."""
from __future__ import annotations

import ast
import math
import operator
import re

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "exp": math.exp, "abs": abs, "round": round}


def calculator(expression: str) -> str:
    """Safely evaluate arithmetic (no eval(): walks a whitelisted AST)."""

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"unsupported expression: {ast.dump(n)[:40]}")

    try:
        v = ev(ast.parse(expression.replace("^", "**").replace("×", "*"), mode="eval").body)
        return str(round(v, 6) if isinstance(v, float) else v)
    except Exception as e:  # tool errors are returned to the agent, not raised
        return f"error: {e}"


ARITH = re.compile(r"(?:what is|calculate|compute|evaluate)?\s*([-+*/().\d\s^%×]+|sqrt\([^)]*\))\s*\??$", re.I)


def extract_arithmetic(question: str) -> str | None:
    m = ARITH.search(question.strip())
    if m and re.search(r"\d\s*[-+*/^%×]\s*\d|sqrt\(", m.group(1)):
        return m.group(1).strip()
    return None


def lookup(term: str, passages: list[str], max_hits: int = 3) -> list[str]:
    """Exact-phrase lookup: sentences in the corpus that mention `term`."""
    out = []
    pat = re.compile(r"\b" + re.escape(term) + r"\b", re.I)
    for p in passages:
        for sent in re.split(r"(?<=[.!?])\s+", p):
            if pat.search(sent):
                out.append(sent)
                if len(out) >= max_hits:
                    return out
    return out
