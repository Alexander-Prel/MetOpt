from __future__ import annotations

import argparse
import ast
import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:  # pragma: no cover - зависимости могут отсутствовать в среде
    np = None
    plt = None
    PdfPages = None


ALLOWED_FUNCS: Dict[str, Callable[..., float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "abs": abs,
    "pow": pow,
}

ALLOWED_CONSTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


class SafeExpression:
    """
    Безопасная обработка строковой формулы f(x) с использованием AST.
    Разрешены только базовые арифметические операции и функции из ALLOWED_FUNCS.
    """

    def __init__(self, expr: str) -> None:
        self.expr = self._normalize(expr)
        self._tree = ast.parse(self.expr, mode="eval")
        self._validate(self._tree)

    @staticmethod
    def _normalize(expr: str) -> str:
        expr = expr.strip()
        if expr.lower().startswith("f(x)"):
            if "=" in expr:
                expr = expr.split("=", 1)[1].strip()
        expr = expr.replace("^", "**")
        return expr

    def _validate(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if not isinstance(child.func, ast.Name):
                    raise ValueError("Разрешены только вызовы функций вида sin(x), cos(x), ...")
                if child.func.id not in ALLOWED_FUNCS:
                    raise ValueError(f"Функция '{child.func.id}' не разрешена")
            elif isinstance(child, ast.Name):
                if child.id not in ("x", *ALLOWED_CONSTS.keys(), *ALLOWED_FUNCS.keys()):
                    raise ValueError(f"Имя '{child.id}' не разрешено")
            elif isinstance(
                child,
                (
                    ast.Expression,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.Pow,
                    ast.Mod,
                    ast.USub,
                    ast.UAdd,
                    ast.Constant,
                    ast.Load,
                ),
            ):
                continue
            else:
                raise ValueError(f"Недопустимый элемент выражения: {type(child).__name__}")

    def _eval(self, node: ast.AST, x: float) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Разрешены только числовые константы")
        if isinstance(node, ast.Name):
            if node.id == "x":
                return float(x)
            return float(ALLOWED_CONSTS[node.id])
        if isinstance(node, ast.UnaryOp):
            val = self._eval(node.operand, x)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
            raise ValueError("Недопустимая унарная операция")
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left, x)
            right = self._eval(node.right, x)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.Mod):
                return left % right
            raise ValueError("Недопустимая бинарная операция")
        if isinstance(node, ast.Call):
            func = ALLOWED_FUNCS[node.func.id]
            args = [self._eval(arg, x) for arg in node.args]
            return float(func(*args))
        raise ValueError("Недопустимое выражение")

    def __call__(self, x: float) -> float:
        return float(self._eval(self._tree.body, x))


@dataclass
class PiyavskiiResult:
    x_best: float
    f_best: float
    iterations: int
    elapsed_sec: float
    xs: List[float]
    fs: List[float]
    L_used: float
    p_min: float
    gap: float


def estimate_lipschitz(xs: List[float], fs: List[float], r: float) -> float:
    max_slope = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        if dx != 0:
            max_slope = max(max_slope, abs((fs[i] - fs[i - 1]) / dx))
    if max_slope == 0.0:
        max_slope = 1.0
    return r * max_slope


def piyavskii_minimize(
    f: Callable[[float], float],
    a: float,
    b: float,
    eps: float = 1e-2,
    L: Optional[float] = None,
    r: float = 1.5,
    max_iters: int = 10000,
) -> PiyavskiiResult:
    """
    Метод Пиявского (Piyavskii–Shubert) для глобального минимума на [a, b].
    """
    if a > b:
        a, b = b, a

    t0 = time.perf_counter()
    xs = [float(a), float(b)]
    fs = [float(f(a)), float(f(b))]

    best_idx = 0 if fs[0] <= fs[1] else 1
    x_best = xs[best_idx]
    f_best = fs[best_idx]

    it = 0
    p_min_global = float("inf")
    Lcur = float(L) if (L is not None and L > 0) else estimate_lipschitz(xs, fs, r)

    while it < max_iters:
        it += 1

        if L is None or L <= 0:
            Lcur = estimate_lipschitz(xs, fs, r)

        x_new = None
        p_min_global = float("inf")

        for i in range(len(xs) - 1):
            xL, xR = xs[i], xs[i + 1]
            fL, fR = fs[i], fs[i + 1]

            # точка пересечения "конусов" (ломаных снизу)
            x_star = 0.5 * (xL + xR) + (fL - fR) / (2.0 * Lcur)
            x_star = min(max(x_star, xL), xR)

            # значение миноранты на интервале
            p_star = 0.5 * (fL + fR) - 0.5 * Lcur * (xR - xL)

            if p_star < p_min_global:
                p_min_global = p_star
                x_new = x_star

        if x_new is None:
            break

        if (f_best - p_min_global) <= eps:
            break

        if any(abs(x_new - x) < 1e-12 for x in xs):
            break

        f_new = float(f(x_new))

        # вставка точки в отсортированные списки
        pos = 0
        while pos < len(xs) and xs[pos] < x_new:
            pos += 1
        xs.insert(pos, x_new)
        fs.insert(pos, f_new)

        if f_new < f_best:
            f_best = f_new
            x_best = x_new

    elapsed = time.perf_counter() - t0
    gap = f_best - p_min_global
    return PiyavskiiResult(
        x_best=x_best,
        f_best=f_best,
        iterations=it,
        elapsed_sec=elapsed,
        xs=xs,
        fs=fs,
        L_used=Lcur,
        p_min=p_min_global,
        gap=gap,
    )


def build_minorant(xs: List[float], fs: List[float], L: float, grid: "np.ndarray") -> "np.ndarray":
    p = np.full_like(grid, -np.inf, dtype=float)
    for xi, fi in zip(xs, fs):
        p = np.maximum(p, fi - L * np.abs(grid - xi))
    return p


def plot_results(
    f: Callable[[float], float],
    res: PiyavskiiResult,
    a: float,
    b: float,
    expr: str,
    eps: float,
    pdf_name: Optional[str] = "demo_report.pdf",
    png_name: Optional[str] = "demo_plot.png",
) -> None:
    if np is None or plt is None:
        raise RuntimeError("Для визуализации нужны numpy и matplotlib")

    grid = np.linspace(a, b, 2000)
    y = np.array([f(x) for x in grid])
    p = build_minorant(res.xs, res.fs, res.L_used, grid)
    poly = np.interp(grid, res.xs, res.fs)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(grid, y, label="f(x)")
    plt.plot(grid, p, label="миноранта p(x)")
    plt.plot(grid, poly, "--", label="ломаная по пробным точкам")
    plt.scatter(res.xs, res.fs, s=18, label="пробные точки")
    plt.scatter([res.x_best], [res.f_best], s=90, marker="*", label="найденный минимум")
    plt.axvline(res.x_best, linestyle="--", linewidth=1)
    plt.title("Метод Пиявского: глобальный минимум на отрезке")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()

    if png_name:
        plt.savefig(png_name, dpi=200, bbox_inches="tight")

    if pdf_name and PdfPages is not None:
        with PdfPages(pdf_name) as pdf:
            pdf.savefig(fig, bbox_inches="tight")

            fig2 = plt.figure(figsize=(8.27, 11.69))
            plt.axis("off")
            text = "\n".join(
                [
                    "Демонстрация: метод Пиявского (Piyavskii–Shubert)",
                    "",
                    f"Функция: f(x) = {expr}",
                    f"Отрезок: [{a}, {b}]",
                    "",
                    f"eps = {eps:.6f}",
                    "",
                    f"x* ≈ {res.x_best:.6f}",
                    f"f(x*) ≈ {res.f_best:.6f}",
                    "",
                    f"Итераций: {res.iterations}",
                    f"Точек (проб): {len(res.xs)}",
                    f"Время: {res.elapsed_sec:.6f} сек",
                    "",
                    f"Оценка L: {res.L_used:.6f}",
                    f"Критерий: f_best - p_min = {res.gap:.6f}",
                ]
            )
            fig2.text(0.06, 0.95, text, va="top", fontsize=12)
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

    plt.show()
    plt.close(fig)


def run(
    expr: str,
    a: float,
    b: float,
    eps: float,
    L: Optional[float],
    r: float,
    max_iters: int,
    pdf_name: Optional[str],
    png_name: Optional[str],
    do_plot: bool,
) -> PiyavskiiResult:
    f = SafeExpression(expr)
    res = piyavskii_minimize(f, a, b, eps=eps, L=L, r=r, max_iters=max_iters)
    expr_show = f.expr

    print("Результат:")
    print(f"x* = {res.x_best:.8f}")
    print(f"f(x*) = {res.f_best:.8f}")
    print(f"Итераций: {res.iterations}")
    print(f"Точек (проб): {len(res.xs)}")
    print(f"Время: {res.elapsed_sec:.6f} сек")
    print(f"L (использованное): {res.L_used:.6f}")
    print(f"f_best - p_min = {res.gap:.6f}")

    if do_plot:
        if np is None or plt is None:
            print("Визуализация пропущена: не установлены numpy/matplotlib.")
        else:
            plot_results(
                f,
                res,
                a,
                b,
                expr_show,
                eps,
                pdf_name=pdf_name,
                png_name=png_name,
            )

    return res


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Поиск глобального минимума липшицевой функции (метод Пиявского)"
    )
    parser.add_argument("--expr", type=str, help="Формула, например: x + sin(pi*x)")
    parser.add_argument("--a", type=float, help="Левая граница отрезка")
    parser.add_argument("--b", type=float, help="Правая граница отрезка")
    parser.add_argument("--eps", type=float, default=0.01, help="Точность eps")
    parser.add_argument("--L", type=float, default=None, help="Липшицева константа (если известна)")
    parser.add_argument("--r", type=float, default=1.5, help="Коэффициент запаса для оценки L")
    parser.add_argument("--max-iters", type=int, default=10000, help="Максимум итераций")
    parser.add_argument("--pdf", type=str, default="demo_report.pdf", help="Имя PDF отчета")
    parser.add_argument("--png", type=str, default="demo_plot.png", help="Имя PNG графика")
    parser.add_argument("--no-plot", action="store_true", help="Отключить визуализацию")
    parser.add_argument("--interactive", action="store_true", help="Ввод через консоль")
    return parser.parse_args(argv)


def prompt_float(name: str, default: Optional[float] = None) -> float:
    while True:
        raw = input(f"{name} = ").strip()
        if not raw and default is not None:
            return float(default)
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print("Введите число, например 0.01")


def main() -> None:
    args = parse_args(sys.argv[1:])

    if args.expr is None:
        if args.interactive or sys.stdin.isatty():
            expr = input("f(x) = ").strip()
            a = prompt_float("a")
            b = prompt_float("b")
            eps = prompt_float("eps", args.eps)
        else:
            expr = "10 + x*x - 10*cos(2*pi*x)"
            a, b, eps = -5.12, 5.12, args.eps
    else:
        expr = args.expr
        a = args.a
        b = args.b
        eps = args.eps

    if a is None or b is None:
        raise ValueError("Нужно задать границы отрезка a и b")

    run(
        expr=expr,
        a=a,
        b=b,
        eps=eps,
        L=args.L,
        r=args.r,
        max_iters=args.max_iters,
        pdf_name=None if args.no_plot else args.pdf,
        png_name=None if args.no_plot else args.png,
        do_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
