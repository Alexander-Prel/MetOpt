from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Tuple, Dict, Optional
import sys


@dataclass
class LPProblem:
    sense: str
    var_names: List[str]
    c: List[Fraction]
    A: List[List[Fraction]]
    rel: List[str]
    b: List[Fraction]


@dataclass
class LPSolution:
    status: str
    x: Dict[str, Fraction]
    objective: Fraction
    reason: str


def _to_frac(s: str) -> Fraction:
    s = s.strip()
    if "/" in s:
        a, b = s.split("/")
        return Fraction(int(a.strip()), int(b.strip()))
    if "." in s:
        return Fraction(s)
    return Fraction(int(s))


def read_lp_from_file(path: str) -> LPProblem:
    # парсим входной файл
    with open(path, "r", encoding="utf-8") as f:
        lines = []
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            if "#" in raw:
                raw = raw.split("#", 1)[0].strip()
            if raw:
                lines.append(raw)

    sense = None
    var_names = None
    c = None
    A: List[List[Fraction]] = []
    rel: List[str] = []
    b: List[Fraction] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        low = line.lower()
        if low.startswith("sense:"):
            sense = line.split(":", 1)[1].strip().lower()
            if sense not in ("min", "max"):
                raise ValueError("sense должен быть min или max")
        elif low.startswith("vars:"):
            var_names = line.split(":", 1)[1].strip().split()
        elif low.startswith("obj:"):
            parts = line.split(":", 1)[1].strip().split()
            c = [_to_frac(p) for p in parts]
        elif low.startswith("constr:"):
            i += 1
            while i < len(lines):
                row = lines[i].strip()
                low2 = row.lower()
                if low2.startswith(("sense:", "vars:", "obj:", "constr:")):
                    i -= 1
                    break

                toks = row.split()
                sign_pos = None
                for k, t in enumerate(toks):
                    if t in ("<=", ">=", "="):
                        sign_pos = k
                        break
                if sign_pos is None:
                    raise ValueError(f"Не найден знак <=, >= или = в строке: {row}")

                coeffs = toks[:sign_pos]
                sign = toks[sign_pos]
                rhs = toks[sign_pos + 1:]
                if len(rhs) != 1:
                    raise ValueError(f"Плохая правая часть в строке: {row}")

                A.append([_to_frac(x) for x in coeffs])
                rel.append(sign)
                b.append(_to_frac(rhs[0]))
                i += 1
        else:
            raise ValueError(f"Неизвестная строка: {line}")
        i += 1

    if sense is None or var_names is None or c is None:
        raise ValueError("В файле должны быть sense, vars, obj и constr")

    if len(c) != len(var_names):
        raise ValueError("Число коэффициентов obj должно совпадать с числом vars")

    for r in A:
        if len(r) != len(var_names):
            raise ValueError("В каждой строке constr должно быть столько коэффициентов, сколько переменных")

    return LPProblem(sense=sense, var_names=var_names, c=c, A=A, rel=rel, b=b)

def _fmt_frac(f: Fraction) -> str:
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def _print_table(T: List[List[Fraction]],
                 basis: List[int],
                 col_names: List[str],
                 title: str,
                 phase: str,
                 it: int) -> None:
    # печать симплекс-таблицы
    m = len(T) - 1
    header = ["Базис"] + col_names + ["RHS"]
    widths = [max(5, len("Базис"))] + [max(3, len(nm)) for nm in col_names] + [max(3, len("RHS"))]

    rows_txt = []

    for i in range(m):
        bname = col_names[basis[i]] if 0 <= basis[i] < len(col_names) else "?"
        row = [bname] + [_fmt_frac(x) for x in T[i][:-1]] + [_fmt_frac(T[i][-1])]
        rows_txt.append(row)

        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    obj = ["rc"] + [_fmt_frac(x) for x in T[-1][:-1]] + [_fmt_frac(T[-1][-1])]
    for j, cell in enumerate(obj):
        widths[j] = max(widths[j], len(cell))

    def line_sep() -> str:
        return "+".join(["-" * (w + 2) for w in widths])

    def fmt_row(r: List[str]) -> str:
        return " | ".join([r[j].rjust(widths[j]) for j in range(len(r))])

    print("\n" + "=" * 90)
    print(f"{title} | {phase} | итерация {it}")
    print("=" * 90)
    print(fmt_row(header))
    print(line_sep())
    for r in rows_txt:
        print(fmt_row(r))
    print(line_sep())
    print(fmt_row(obj))
    print("=" * 90)


def _pivot(T: List[List[Fraction]], row: int, col: int) -> None:
    piv = T[row][col]
    if piv == 0:
        raise ZeroDivisionError("Разрешающий элемент 0")

    # нормируем строку и зануляем столбец
    T[row] = [v / piv for v in T[row]]

    for r in range(len(T)):
        if r == row:
            continue
        factor = T[r][col]
        if factor != 0:
            T[r] = [T[r][c] - factor * T[row][c] for c in range(len(T[r]))]


def _compute_reduced_costs(T: List[List[Fraction]], basis: List[int], c: List[Fraction]) -> None:
    # считаем reduced costs (min), RHS хранит -z
    m = len(basis)
    n = len(c)

    obj = [Fraction(0) for _ in range(n + 1)]

    for j in range(n):
        s = Fraction(0)
        for i in range(m):
            s += c[basis[i]] * T[i][j]
        obj[j] = c[j] - s

    s = Fraction(0)
    for i in range(m):
        s += c[basis[i]] * T[i][-1]
    obj[-1] = Fraction(0) - s

    T[-1] = obj


def _choose_entering_col_bland(T: List[List[Fraction]]) -> Optional[int]:
    last = T[-1]
    for j in range(len(last) - 1):
        if last[j] < 0:
            return j
    return None


def _choose_leaving_row_bland(T: List[List[Fraction]], col: int) -> Optional[int]:
    best_i = None
    best_ratio = None
    for i in range(len(T) - 1):
        a = T[i][col]
        b = T[i][-1]
        if a > 0:
            ratio = b / a
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_i = i
    return best_i


def _simplex_min(T: List[List[Fraction]],
                 basis: List[int],
                 c: List[Fraction],
                 col_names: List[str],
                 phase: str,
                 verbose: bool = True,
                 max_iters: int = 20000) -> Tuple[str, str]:
    # итерации симплекса для минимума
    _compute_reduced_costs(T, basis, c)
    if verbose:
        _print_table(T, basis, col_names, title="Симплекс-таблица", phase=phase, it=0)

    for it in range(1, max_iters + 1):
        enter = _choose_entering_col_bland(T)
        if enter is None:
            return "optimal", "Оптимум: все reduced costs >= 0 (для min)."

        leave = _choose_leaving_row_bland(T, enter)
        if leave is None:
            return "unbounded", "Целевая не ограничена снизу (unbounded)."

        if verbose:
            print(f"\nШаг: вводим в базис столбец '{col_names[enter]}', выводим строку {leave + 1}")

        basis[leave] = enter
        _pivot(T, leave, enter)
        _compute_reduced_costs(T, basis, c)

        if verbose:
            _print_table(T, basis, col_names, title="Симплекс-таблица", phase=phase, it=it)

    return "unbounded", "Превышено число итераций (возможна вырожденность/зацикливание)."


def solve_lp(problem: LPProblem, verbose: bool = True) -> LPSolution:
    # канонизация и запуск двух фаз
    is_max = (problem.sense == "max")
    c_orig = [Fraction(x) for x in problem.c]
    c_min = [-x for x in c_orig] if is_max else [Fraction(x) for x in c_orig]

    m = len(problem.A)
    n = len(problem.var_names)

    # нормализация RHS
    A = [row[:] for row in problem.A]
    rel = problem.rel[:]
    b = [Fraction(x) for x in problem.b]

    for i in range(m):
        if b[i] < 0:
            b[i] = -b[i]
            A[i] = [-Fraction(x) for x in A[i]]
            if rel[i] == "<=":
                rel[i] = ">="
            elif rel[i] == ">=":
                rel[i] = "<="

    # расширяем матрицу ограничений
    col_names = problem.var_names[:]
    coeff = [[Fraction(x) for x in A[i]] for i in range(m)]
    rhs = [Fraction(x) for x in b]

    def add_col_end(name: str, values: List[Fraction]) -> int:
        col_names.append(name)
        for i in range(m):
            coeff[i].append(values[i])
        return len(col_names) - 1

    artificial_cols: List[int] = []
    basis: List[int] = [-1] * m
    slack_count = 0
    art_count = 0

    for i in range(m):
        if rel[i] == "<=":
            slack_count += 1
            sname = f"s{slack_count}"
            col = [Fraction(0)] * m
            col[i] = Fraction(1)
            s_idx = add_col_end(sname, col)
            basis[i] = s_idx

        elif rel[i] == ">=":
            slack_count += 1
            sname = f"s{slack_count}"
            col_s = [Fraction(0)] * m
            col_s[i] = Fraction(-1)
            add_col_end(sname, col_s)

            art_count += 1
            aname = f"a{art_count}"
            col_a = [Fraction(0)] * m
            col_a[i] = Fraction(1)
            a_idx = add_col_end(aname, col_a)
            artificial_cols.append(a_idx)
            basis[i] = a_idx

        elif rel[i] == "=":
            art_count += 1
            aname = f"a{art_count}"
            col_a = [Fraction(0)] * m
            col_a[i] = Fraction(1)
            a_idx = add_col_end(aname, col_a)
            artificial_cols.append(a_idx)
            basis[i] = a_idx

        else:
            return LPSolution("infeasible", {}, Fraction(0), f"Неизвестный знак: {rel[i]}")

    T = [coeff[i] + [rhs[i]] for i in range(m)]
    T.append([Fraction(0) for _ in range(len(col_names))] + [Fraction(0)])

    # фаза 1
    if artificial_cols:
        if verbose:
            print("\n\n########## ФАЗА 1: ВСПОМОГАТЕЛЬНАЯ ЗАДАЧА min w = сумма искусственных ##########")

        c_phase1 = [Fraction(0) for _ in range(len(col_names))]
        for j in artificial_cols:
            c_phase1[j] = Fraction(1)

        status1, reason1 = _simplex_min(T, basis, c_phase1, col_names, phase="Фаза 1", verbose=verbose)
        if status1 != "optimal":
            return LPSolution(status1, {}, Fraction(0), reason1)

        w_opt = -T[-1][-1]
        if verbose:
            print(f"\nРезультат Фазы 1: w* = {_fmt_frac(w_opt)}")

        if w_opt != 0:
            return LPSolution("infeasible", {}, w_opt, f"Решений нет: во вспомогательной задаче w*={w_opt} > 0.")

        for i in range(m):
            if basis[i] in artificial_cols:
                for j in range(len(col_names)):
                    if j not in artificial_cols and T[i][j] != 0:
                        basis[i] = j
                        _pivot(T, i, j)
                        break

        # удаляем artificial-столбцы
        keep = [j for j in range(len(col_names)) if j not in artificial_cols]
        new_names = [col_names[j] for j in keep]

        T2 = []
        for r in range(len(T)):
            T2.append([T[r][j] for j in keep] + [T[r][-1]])

        old_to_new = {old: new for new, old in enumerate(keep)}
        basis = [old_to_new[basis[i]] for i in range(m)]
        col_names = new_names
        T = T2

    # фаза 2
    if verbose:
        print("\n\n########## ФАЗА 2: ОСНОВНАЯ ЗАДАЧА ##########")

    c_phase2 = [Fraction(0) for _ in range(len(col_names))]
    name_to_idx = {nm: i for i, nm in enumerate(col_names)}
    for j, xname in enumerate(problem.var_names):
        c_phase2[name_to_idx[xname]] = c_min[j]

    status2, reason2 = _simplex_min(T, basis, c_phase2, col_names, phase="Фаза 2", verbose=verbose)
    if status2 != "optimal":
        return LPSolution(status2, {}, Fraction(0), reason2)

    z_min_opt = -T[-1][-1]
    z_orig = -z_min_opt if is_max else z_min_opt

    x_vals = {nm: Fraction(0) for nm in col_names}
    for i in range(m):
        x_vals[col_names[basis[i]]] = T[i][-1]

    x_main = {nm: x_vals.get(nm, Fraction(0)) for nm in problem.var_names}

    return LPSolution("optimal", x_main, z_orig, "ОК")


def main():
    # точка входа CLI
    if len(sys.argv) < 2:
        print("Использование: python lp_universal_simplex_verbose.py input.txt [--quiet]")
        sys.exit(1)

    path = sys.argv[1]
    verbose = True
    if len(sys.argv) >= 3 and sys.argv[2] == "--quiet":
        verbose = False

    prob = read_lp_from_file(path)
    sol = solve_lp(prob, verbose=verbose)

    print("\n\n================== ИТОГ ==================")
    print("СТАТУС:", sol.status)
    if sol.status == "optimal":
        print("Оптимальная точка:")
        for k in prob.var_names:
            print(f"  {k} = {_fmt_frac(sol.x[k])}")
        print("Значение целевой функции:", _fmt_frac(sol.objective))
    else:
        print("Причина:", sol.reason)


if __name__ == "__main__":
    main()
