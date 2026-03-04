from __future__ import annotations

import math
import os
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# Streamlit "not updating" fix:
# Clear session_state only ONCE per browser session.
# (Clearing on every rerun breaks inputs.)
# ============================================================
if "___cleared_once" not in st.session_state:
    st.session_state.clear()
    st.session_state["___cleared_once"] = True

# -------------------------
# Model parameters
# -------------------------
PRODUCTS = ["Rolls", "Croissant", "Cake"]


@dataclass
class Product:
    price: float
    unit_cost: float
    salvage: float      # revenue per leftover item (e.g., discounted sale)
    waste_cost: float   # disposal/quality cost per leftover item


@dataclass
class Costs:
    fixed_cost_per_day: float
    extra_staff_cost: float


# Seasonal demand: base daily Poisson lambda per month
MONTH_LAMBDA = {
    1: 220, 2: 225, 3: 245, 4: 265, 5: 280, 6: 295,
    7: 285, 8: 275, 9: 295, 10: 285, 11: 280, 12: 340
}
DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}

DEFAULT_PRODUCTS: Dict[str, Product] = {
    "Rolls":     Product(price=0.55, unit_cost=0.18, salvage=0.10, waste_cost=0.02),
    "Croissant": Product(price=1.60, unit_cost=0.60, salvage=0.35, waste_cost=0.03),
    "Cake":      Product(price=3.20, unit_cost=1.25, salvage=0.80, waste_cost=0.05),
}


def poisson_knuth(lmbda: float, rng: random.Random) -> int:
    if lmbda <= 0:
        return 0
    L = math.exp(-lmbda)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def basket_probs(month: int) -> Dict[str, float]:
    # simple seasonal preference shifts (winter more cake)
    p = {"Rolls": 0.62, "Croissant": 0.23, "Cake": 0.15}
    if month in (11, 12, 1, 2):
        p["Cake"] += 0.06
        p["Rolls"] -= 0.04
        p["Croissant"] -= 0.02
    s = sum(p.values())
    return {k: v / s for k, v in p.items()}


def expected_items_per_customer(month: int) -> float:
    # expected value of the SAME distribution used in the demand model
    weights = [0.25, 0.40, 0.25, 0.10] if month == 12 else [0.30, 0.42, 0.22, 0.06]
    return 1 * weights[0] + 2 * weights[1] + 3 * weights[2] + 4 * weights[3]


def simulate_one_year(
    rng: random.Random,
    products: Dict[str, Product],
    costs: Costs,
    production_plan: Dict[str, int],   # treated as base MAX capacity per day
    extra_staff: int,
    demand_noise_sd: float,
    capacity_gain_per_staff: float,    # ✅ NEW: capacity boost per extra staff
    safety: float,                     # adaptive production safety factor
) -> Tuple[float, List[float], Dict[str, float]]:
    monthly_profit: List[float] = []
    total_revenue = total_var = total_fixed = total_salv = total_waste = 0.0
    total_stockout = total_waste_units = 0

    for m in range(1, 13):
        month_profit = 0.0
        probs = basket_probs(m)
        e_items = expected_items_per_customer(m)

        # Precompute weights once per month (speed)
        prod_weights = [probs["Rolls"], probs["Croissant"], probs["Cake"]]
        item_weights = [0.25, 0.40, 0.25, 0.10] if m == 12 else [0.30, 0.42, 0.22, 0.06]
        item_values = [1, 2, 3, 4]

        for _ in range(DAYS_IN_MONTH[m]):
            # demand level for THIS day
            noise_mult = max(0.4, min(1.6, 1.0 + rng.gauss(0.0, demand_noise_sd)))
            lam = MONTH_LAMBDA[m] * noise_mult

            # -----------------------------------------
            # Adaptive production: decide what to produce today
            # (plan is treated as base capacity; staff can increase capacity)
            # -----------------------------------------
            expected_total_items = lam * e_items
            produced_today: Dict[str, int] = {}

            for pname in PRODUCTS:
                base_cap = int(production_plan.get(pname, 0))
                cap = int(round(base_cap * (1.0 + extra_staff * capacity_gain_per_staff)))  # ✅ staff increases cap
                exp_units = expected_total_items * probs[pname]
                produced_today[pname] = min(cap, max(0, int(round(exp_units * safety))))

            # realized demand
            customers = poisson_knuth(lam, rng)

            # FAST demand sampling (batch)
            if customers > 0:
                items_list = rng.choices(item_values, weights=item_weights, k=customers)
                total_items = sum(items_list)

                chosen = rng.choices(PRODUCTS, weights=prod_weights, k=total_items)
                c = Counter(chosen)
                demand_units = {
                    "Rolls": c.get("Rolls", 0),
                    "Croissant": c.get("Croissant", 0),
                    "Cake": c.get("Cake", 0),
                }
            else:
                demand_units = {"Rolls": 0, "Croissant": 0, "Cake": 0}

            revenue = var_cost = salvage = waste_cost = 0.0
            stockout_units = waste_units = 0

            for pname in PRODUCTS:
                produced = int(produced_today[pname])
                demanded = int(demand_units[pname])
                sold = min(produced, demanded)
                stockout = max(0, demanded - produced)
                waste = max(0, produced - demanded)

                p = products[pname]
                revenue += sold * p.price
                var_cost += sold * p.unit_cost       # ✅ variable cost on SOLD items
                salvage += waste * p.salvage
                waste_cost += waste * p.waste_cost

                stockout_units += stockout
                waste_units += waste

            fixed = costs.fixed_cost_per_day + extra_staff * costs.extra_staff_cost
            profit_day = (revenue + salvage) - (var_cost + waste_cost + fixed)
            month_profit += profit_day

            total_revenue += revenue
            total_var += var_cost
            total_fixed += fixed
            total_salv += salvage
            total_waste += waste_cost
            total_stockout += stockout_units
            total_waste_units += waste_units

        monthly_profit.append(month_profit)

    kpis = {
        "revenue": total_revenue,
        "variable_cost": total_var,
        "fixed_cost": total_fixed,
        "salvage": total_salv,
        "waste_cost": total_waste,
        "stockout_units": float(total_stockout),
        "waste_units": float(total_waste_units),
    }
    return sum(monthly_profit), monthly_profit, kpis


def run_mc(
    runs: int,
    seed: int,
    products: Dict[str, Product],
    costs: Costs,
    production_plan: Dict[str, int],
    extra_staff: int,
    demand_noise_sd: float,
    capacity_gain_per_staff: float,
    safety: float,
):
    rng = random.Random(seed)
    profits: List[float] = []
    monthly_avg = [0.0] * 12
    stockouts: List[float] = []

    for _ in range(runs):
        p_year, p_month, kpi = simulate_one_year(
            rng=rng,
            products=products,
            costs=costs,
            production_plan=production_plan,
            extra_staff=extra_staff,
            demand_noise_sd=demand_noise_sd,
            capacity_gain_per_staff=capacity_gain_per_staff,
            safety=safety,
        )
        profits.append(p_year)
        stockouts.append(kpi["stockout_units"])
        for j in range(12):
            monthly_avg[j] += p_month[j]

    monthly_avg = [x / runs for x in monthly_avg]
    return {
        "mean_profit": statistics.mean(profits),
        "stdev_profit": statistics.pstdev(profits) if runs > 1 else 0.0,
        "mean_stockout": statistics.mean(stockouts),
        "profits": profits,
        "monthly_avg": monthly_avg,
    }


def plot_monthly(monthly_avg: List[float]):
    fig = plt.figure()
    x = list(range(1, 13))
    plt.plot(x, monthly_avg, marker="o")
    plt.title("Average profit per month")
    plt.xlabel("Month")
    plt.ylabel("Profit [€]")
    plt.xticks(x)
    plt.tight_layout()
    return fig


def staff_sweep(
    runs: int,
    seed: int,
    staff_min: int,
    staff_max: int,
    penalty_per_stockout: float,
    products: Dict[str, Product],
    costs: Costs,
    production_plan: Dict[str, int],
    demand_noise_sd: float,
    capacity_gain_per_staff: float,
    safety: float,
):
    out: List[Tuple[int, float, float, float]] = []
    for n in range(staff_min, staff_max + 1):
        res = run_mc(
            runs=runs,
            seed=seed + n * 1337,
            products=products,
            costs=costs,
            production_plan=production_plan,
            extra_staff=n,
            demand_noise_sd=demand_noise_sd,
            capacity_gain_per_staff=capacity_gain_per_staff,
            safety=safety,
        )
        score = res["mean_profit"] - penalty_per_stockout * res["mean_stockout"]
        out.append((n, res["mean_profit"], res["mean_stockout"], score))
    return out


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="MC Bakery", layout="wide")
st.title("Monte-Carlo Simulation: Bakery")
st.caption(f"Running: {os.path.abspath(__file__)}")

with st.sidebar:
    st.header("Inputs")

    # reset widget values (for keys ending in _v7)
    if st.button("Reset inputs"):
        for k in list(st.session_state.keys()):
            if k.endswith("_v7"):
                del st.session_state[k]
        st.rerun()

    runs = st.number_input("Runs", min_value=10, max_value=250, value=120, step=10, key="runs_v7")
    seed = st.number_input("Seed", min_value=0, max_value=300, value=42, step=1, key="seed_v7")

    st.subheader("Production plan (BASE max capacity/day)")
    prod_rolls = st.number_input("Rolls/day", min_value=0, max_value=200, value=150, step=10, key="rolls_v7")
    prod_croissant = st.number_input("Croissant/day", min_value=0, max_value=150, value=100, step=5, key="croissant_v7")
    prod_cake = st.number_input("Cake/day", min_value=0, max_value=100, value=60, step=5, key="cake_v7")

    st.subheader("Costs")
    fixed_cost = st.number_input("Fixed cost/day [€]", min_value=0.0, max_value=200.0, value=90.0, step=10.0, key="fixed_v7")
    staff_cost = st.number_input("Extra staff cost/day [€]", min_value=0.0, max_value=400.0, value=20.0, step=5.0, key="staff_v7")

    st.subheader("Demand / Production policy")
    demand_noise_sd = st.slider("Demand noise SD", 0.0, 0.5, 0.12, 0.01, key="noise_v7")
    safety = st.slider("Safety factor (produce vs expected demand)", 0.80, 1.20, 1.05, 0.01, key="safety_v7")

    st.subheader("Staff effect (IMPORTANT)")
    capacity_gain_per_staff = st.slider(
        "Capacity gain per extra staff (e.g. 0.08 = +8% per staff)",
        0.0, 0.30, 0.08, 0.01,
        key="capgain_v7"
    )

products = DEFAULT_PRODUCTS
costs = Costs(fixed_cost_per_day=float(fixed_cost), extra_staff_cost=float(staff_cost))
plan = {"Rolls": int(prod_rolls), "Croissant": int(prod_croissant), "Cake": int(prod_cake)}

tab1, tab2 = st.tabs(["Simulation", "Optimal staffing"])

with tab1:
    extra_staff = st.number_input(
        "Extra staff (for this run)",
        min_value=0, max_value=50, value=1, step=1,
        key="extrastaff_v7"
    )
    if st.button("Run simulation"):
        res = run_mc(
            runs=int(runs),
            seed=int(seed),
            products=products,
            costs=costs,
            production_plan=plan,
            extra_staff=int(extra_staff),
            demand_noise_sd=float(demand_noise_sd),
            capacity_gain_per_staff=float(capacity_gain_per_staff),
            safety=float(safety),
        )
        st.write({
            "E[Yearly profit]": round(res["mean_profit"], 2),
            "σ(profit)": round(res["stdev_profit"], 2),
            "E[Stockout units]": round(res["mean_stockout"], 1),
        })
        st.pyplot(plot_monthly(res["monthly_avg"]))

with tab2:
    staff_min = st.number_input("Staff min", 0, 50, 0, 1)
    staff_max = st.number_input("Staff max", 0, 50, 8, 1)
    penalty = st.number_input("Penalty €/stockout unit", 0.0, 10.0, 0.35, 0.05)

    if st.button("Compute optimal staff"):
        data = staff_sweep(
            runs=int(runs),
            seed=int(seed),
            staff_min=int(staff_min),
            staff_max=int(staff_max),
            penalty_per_stockout=float(penalty),
            products=products,
            costs=costs,
            production_plan=plan,
            demand_noise_sd=float(demand_noise_sd),
            capacity_gain_per_staff=float(capacity_gain_per_staff),
            safety=float(safety),
        )

        best = max(data, key=lambda x: x[3])
        st.success(f"Optimal extra staff N = {best[0]}")

        st.dataframe(
            [{"N": n, "E_Profit": round(mp, 2), "E_Stockout": round(ms, 1), "Score": round(sc, 2)}
             for (n, mp, ms, sc) in data],
            use_container_width=True
        )