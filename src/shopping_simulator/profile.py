from shopping_simulator.simulator import LossSimulation
import pstats
import cProfile
import pandas as pd


class LossSimulationProfile:
    def __init__(
        self, simulation: LossSimulation, total_days: int, stockout_threshold: float
    ):
        self.simulation = simulation
        self.total_days = total_days
        self.stockout_threshold = stockout_threshold

    def run_profile(self, filename: str) -> None:
        sim = self.simulation
        total_days = self.total_days
        stockout_threshold = self.stockout_threshold
        print(sim, total_days, stockout_threshold)
        cProfile.run("sim.calculate_loss(total_days, stockout_threshold)", filename)

    def get_function_location(self, filename):
        try:
            return "/".join(filename.split("/")[-2:])
        except Exception:
            return filename

    def get_stats(self, filename: str) -> pd.DataFrame:
        stats = pstats.Stats(filename)

        stats_list = []
        # stats.print_stats()
        for func in stats.stats:
            cc, nc, tt, ct, callers = stats.stats[func]
            filename, lineno, func_name = func
            stats_list.append(
                {
                    "filename": filename,
                    "line_no": lineno,
                    "function_name": func_name,
                    "call_count": cc,
                    "total_time": tt,
                    "cumulative_time": ct,
                    "callers": callers,
                    "per_call_time": tt / cc if cc else 0,
                    "cumulative_per_call": ct / cc if cc else 0,
                }
            )

        df = pd.DataFrame(stats_list)

        df["location"] = df["filename"].apply(self.get_function_location)
        df["own_function"] = df["filename"].apply(
            lambda x: True if "shoppingsimulator" in x else False
        )
        return df.sort_values("total_time", ascending=False)
