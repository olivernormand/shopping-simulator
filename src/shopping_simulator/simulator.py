"""
Written by Oliver Normand (oliver@normand.uk).

The initial work was done as part of an assessment, this led to scrappy code.

The code was completely rewritten to be more general and optimised. Subsequent additions were since made.
"""

import numpy as np
from scipy.stats import poisson as ps
import functools
from shopping_simulator.models import MinimumLossResult, SimulationResult


def poisson(x, mean):
    return ps.pmf(x, mean)


class LossSimulation:
    def __init__(
        self,
        codelife: int,
        unit_sales_per_day: int,
        units_per_case: int,
        lead_time: int,
        seed: int | None = None,
    ):
        """The product characteristics are defined by 4 parameters. Knowing these enables us to forecast the performance of this product."""
        self.codelife = codelife
        self.unit_sales_per_day = unit_sales_per_day
        self.units_per_case = units_per_case
        self.lead_time = lead_time

        if seed is not None:
            np.random.seed(seed)

        # Initialise the inventory. This is a list of (n_units, days_left) lists where n_units is the number of units in the case and days_left is the number of days remaining before the code expires.
        self.n_units, self.days_left = self.initialise_inventory()

    def initialise_inventory(self):
        n_units = (
            np.ones(1, dtype=int)
            * int(self.codelife * self.unit_sales_per_day / self.units_per_case)
            * self.units_per_case
        )
        days_left = np.ones(1, dtype=int) * self.codelife
        return n_units, days_left

    def calculate_min_loss(
        self,
        total_days: int,
        rotation: bool = True,
        stockout_threshold=np.linspace(1e-6, 0.999, 10),
        multipliers=(1, 1),
        deep_search=0,
        verbose=False,
    ) -> MinimumLossResult:
        """Calculate the minimum loss for a given set of parameters.
        First searching a full range of stockout thresholds. Optionally narrow the search window and repeat.

        The second search considers the stockout threshold that gave the lowest in the first search, and searches a narrower window around that threshold.

        Parameters
        ----------
        total_days : int
            How many days the simulation should run for in total. This is split amongst the searches.
        rotation : bool, optional
            Whether to use stock rotation, by default True
        stockout_threshold : np.ndarray, optional
            What the starting stockout threshold search should be, by default np.linspace(1e-6, 0.999, 10)
        multipliers : tuple, optional
            The weighting of total days between the first and second search, by default (1, 1)
        deep_search : int, optional
            How many search spacings to allocate for the deep search. If zero then no secondary search occurs, by default 0

        Returns
        -------
        MinimumLossResult
            The minimum loss and the stockout threshold that achieves it.
        """

        granularitys = (len(stockout_threshold), deep_search)

        # Total days is the total days available to calculate the min_loss. We scale this down originally.
        if deep_search > 0:
            total_days = int(
                total_days
                / (granularitys[0] * multipliers[0] + granularitys[1] * multipliers[1])
            )
        else:
            total_days = int(total_days / (granularitys[0] * multipliers[0]))

        if verbose:
            print(
                "Before the multiplier, each simulation will run for",
                total_days,
                "days",
            )

        loss = np.zeros(len(stockout_threshold))
        availability_loss = np.zeros(len(stockout_threshold))
        waste_loss = np.zeros(len(stockout_threshold))
        units_sold = np.zeros(len(stockout_threshold))
        eod_units = np.zeros(len(stockout_threshold))

        for i, threshold in enumerate(stockout_threshold):
            (
                loss[i],
                availability_loss[i],
                waste_loss[i],
                units_sold[i],
                eod_units[i],
            ) = self.calculate_loss(
                int(total_days * multipliers[0]),
                threshold,
                rotation=rotation,
                verbose=0,
            )

        stockout_thresholds = [stockout_threshold]
        loss = [loss]
        availability_loss = [availability_loss]
        waste_loss = [waste_loss]
        units_sold = [units_sold]
        eod_units = [eod_units]

        if verbose:
            print("INITIAL SEARCH COMPLETE")
            print("Stockout Thresholds:", stockout_threshold)
            print("Min Loss:", loss)

        if deep_search > 0:
            # Narrow down the search window and go again
            loss_sum = np.convolve(loss[0], np.ones(2), "valid")
            arg_min = np.argmin(loss_sum)
            lower_bound = stockout_threshold[arg_min]
            upper_bound = stockout_threshold[arg_min + 1]
            expand_bound = (upper_bound - lower_bound) / 2
            lower_bound = np.max([0, lower_bound - expand_bound])
            upper_bound = np.min([1, upper_bound + expand_bound])
            stockout_threshold = np.linspace(lower_bound, upper_bound, granularitys[1])

            deep_loss = np.zeros(len(stockout_threshold))
            deep_availability_loss = np.zeros(len(stockout_threshold))
            deep_waste_loss = np.zeros(len(stockout_threshold))
            deep_units_sold = np.zeros(len(stockout_threshold))
            deep_eod_units = np.zeros(len(stockout_threshold))

            for i, threshold in enumerate(stockout_threshold):
                (
                    deep_loss[i],
                    deep_availability_loss[i],
                    deep_waste_loss[i],
                    deep_units_sold[i],
                    deep_eod_units[i],
                ) = self.calculate_loss(
                    int(total_days * multipliers[1]),
                    threshold,
                    rotation=rotation,
                    verbose=0,
                )

            stockout_thresholds.append(stockout_threshold)
            loss.append(deep_loss)
            availability_loss.append(deep_availability_loss)
            waste_loss.append(deep_waste_loss)
            units_sold.append(deep_units_sold)
            eod_units.append(deep_eod_units)

            if verbose:
                print("NARROWED SEARCH WINDOW")
                print("Stockout Thresholds:", stockout_threshold)
                print("Min Loss:", loss)

        stockout_thresholds = np.concatenate(stockout_thresholds, axis=0)
        loss = np.concatenate(loss, axis=0)
        availability_loss = np.concatenate(availability_loss, axis=0)
        waste_loss = np.concatenate(waste_loss, axis=0)
        units_sold = np.concatenate(units_sold, axis=0)
        eod_units = np.concatenate(eod_units, axis=0)

        # Now sort the arrays by the threshold
        idx = np.argsort(stockout_thresholds)
        stockout_thresholds = stockout_thresholds[idx]
        loss = loss[idx]
        availability_loss = availability_loss[idx]
        waste_loss = waste_loss[idx]
        units_sold = units_sold[idx]
        eod_units = eod_units[idx]

        return MinimumLossResult(
            min_loss=np.min(loss),
            min_stockout_threshold=stockout_thresholds[np.argmin(loss)],
            min_waste_loss=waste_loss[np.argmin(loss)],
            min_availability_loss=availability_loss[np.argmin(loss)],
            mean_units_sold_per_day=units_sold[np.argmin(loss)],
            mean_eod_units=eod_units[np.argmin(loss)],
            thresholds=stockout_thresholds,
            loss=loss,
            availability_loss=availability_loss,
            waste_loss=waste_loss,
            units_sold=units_sold,
            eod_units=eod_units,
        )

    def calculate_min_loss_std(
        self,
        total_days: int,
        stockout_threshold: float,
        repeats: int = 10,
        rotation: bool = True,
    ) -> tuple[float, float]:
        """
        Due to the stochastic nature of the simulation, we can't guarantee that the minimum loss we find is actually the minimum loss.
        To account for this, we can run the simulation multiple times and take the mean and standard deviation of the losses.
        This gives us some notion of the variance in the loss.
        """
        losses = np.zeros(repeats)
        for i in range(repeats):
            losses[i], _, _, _, _ = self.calculate_loss(
                total_days // repeats, stockout_threshold, rotation=rotation, verbose=0
            )

        return np.mean(losses), np.std(losses)

    def calculate_loss(
        self, total_days: int, stockout_threshold: float, rotation=True, verbose=0
    ) -> tuple[float, float, float, float, float]:
        """
        Calculate the total loss, availability loss, waste loss, units sold, and eod units for the given parameters.

        Parameters:
        total_days (int): The total number of days to simulate.
        stockout_threshold (int): The threshold for stockout.
        rotation (bool, optional): Whether to consider rotation. Defaults to True.
        verbose (int, optional): The level of verbosity. Defaults to 0.

        Returns:
        tuple: A tuple containing the total loss, availability loss, and waste loss.
        """
        res = self.simulate(
            total_days=total_days,
            stockout_threshold=stockout_threshold,
            rotation=rotation,
            verbose=verbose,
        )

        # Normalise the weights, and apply them to the waste losses.
        availability_loss = np.sum(res.stockout) / len(res.stockout)
        waste_loss = (
            np.sum(res.wasted_units) / np.sum(res.cases_ordered) / self.units_per_case
        )
        units_sold = np.mean(res.units_sold)
        eod_units = np.mean(res.eod_units)

        return (
            availability_loss + waste_loss,
            availability_loss,
            waste_loss,
            units_sold,
            eod_units,
        )

    def simulate(
        self,
        total_days: int = 100,
        stockout_threshold: float = 0.1,
        rotation: bool = True,
        verbose: int = 0,
        n_units: np.ndarray | None = None,
        days_left: np.ndarray | None = None,
    ) -> SimulationResult:
        """Run the simulation to calculate the loss for a given set of parameters.

        Each day do the following:
            1. Calculate the probability of a stockout for the current inventory at lead_time in the future.
                1a. If the probability is below the threshold, then we don't order any more stock.
                1b. If the probability is above the threshold, then we order more stock until the probability is below the threshold. We may need to order more than one case.
            2. Update the inventory by subtracting the units sold today.
            3. At the end of the day:
                3a. Calculate the units wasted as an integer.
                3b. Calculate whether we had a stockout as a boolean.
        """

        if n_units is None:
            n_units = self.n_units
        if days_left is None:
            days_left = self.days_left

        wasted_units = np.zeros(total_days, dtype=int)
        stockout = np.zeros(total_days, dtype=bool)
        cases_ordered = np.zeros(total_days, dtype=int)
        units_sold = np.zeros(total_days, dtype=int)
        eod_units = np.zeros(total_days, dtype=int)

        for day in range(total_days):
            # Each day we calculate the initial probability of a stockout
            prob_stockout = self.probability_stockout(
                tuple(n_units),
                tuple(days_left),
                self.unit_sales_per_day,
                self.lead_time,
                self.codelife,
                rotation,
                verbose,
            )

            if verbose >= 1:
                print()
                print("Day", day, n_units, days_left, prob_stockout)
            if verbose >= 2:
                active_groups = (days_left > 0) & (days_left <= self.codelife)
                print(
                    "Day", day, "Active Groups:", np.argwhere(active_groups).flatten()
                )

            # Order stock, if the threshold is met (if threshold is not met, then these are returned unchanged)
            n_units, days_left, cases_ordered[day] = self.order_stock(
                n_units, days_left, prob_stockout, stockout_threshold, rotation
            )

            # Simulate the impact of the stock on sales
            n_units[days_left <= self.codelife], units_sold[day] = self.simulate_sales(
                n_units[days_left <= self.codelife], rotation, verbose
            )

            eod_units[day] = np.sum(n_units)

            wasted_units_mask = days_left <= 1
            keep_units_mask = (days_left > 1) & (n_units > 0)

            # Calculate the units wasted
            wasted_units[day] = np.sum(n_units[wasted_units_mask])
            # Calculate whether we had a stockout
            stockout[day] = np.sum(n_units[days_left <= self.codelife]) == 0

            if verbose >= 2:
                print("Day", day, "Wasted Units:", wasted_units[day])
                print("Day", day, "Stockout:", stockout[day])

            # Update the inventory
            n_units = n_units[keep_units_mask]
            days_left = days_left[keep_units_mask] - 1

        return SimulationResult(
            wasted_units=wasted_units,
            cases_ordered=cases_ordered,
            stockout=stockout,
            units_sold=units_sold,
            eod_units=eod_units,
        )

    def order_stock(
        self,
        n_units: np.ndarray,
        days_left: np.ndarray,
        prob_stockout: float,
        stockout_threshold: float,
        rotation: bool,
        verbose: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        cases_ordered = 0

        while prob_stockout > stockout_threshold:
            # If we've not met the threshold, then we order a single case and recalculate the probability of a stockout
            # This repeats until the probability of a stockout is below the threshold.
            n_units, days_left = self.order_single_case(n_units, days_left)
            prob_stockout = self.probability_stockout(
                tuple(n_units),
                tuple(days_left),
                self.unit_sales_per_day,
                self.lead_time,
                self.codelife,
                rotation,
                verbose,
            )
            cases_ordered += 1

        return n_units, days_left, cases_ordered

    def order_single_case(
        self, n_units: np.ndarray, days_left: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Update the n_units and days_left arrays to represent an increased stock level.

        If they're currently of length zero, then add a new entry to the end of the arrays.

        If the days_left of the final entry is equal to codelife + lead_time, then add to that inventory level. Otherwise, add a new entry to the end of the arrays.
        """
        if len(n_units) == 0:
            n_units = np.append(n_units, self.units_per_case)
            days_left = np.append(days_left, self.codelife + self.lead_time)

        elif days_left[-1] == self.codelife + self.lead_time:
            n_units[-1] += self.units_per_case

        else:
            n_units = np.append(n_units, self.units_per_case)
            days_left = np.append(days_left, self.codelife + self.lead_time)

        return n_units, days_left

    @functools.lru_cache(maxsize=None)
    def probability_stockout(
        self,
        n_units: np.ndarray,
        days_left: np.ndarray,
        unit_sales_per_day: float,
        lead_time: int,
        codelife: int,
        rotation_weighting: bool = True,
        verbose: int = 0,
    ) -> float:
        """We calculate the probability of a stockout as follows:
        Take in the number of units at each remaining day of codelife, units with remaining codelife that exceeds the shelf_life are aggregated.
        """
        # To enable cacheing, we have passed the arguments of n_units and days_left as tuples.
        # These are subsequently converted back to arrays.

        n_units = np.array(n_units)
        days_left = np.array(days_left)

        assert len(n_units) == len(days_left)

        if rotation_weighting:
            prob_stockout = self.probability_stockout_rotation(
                n_units, days_left, unit_sales_per_day, lead_time, codelife, verbose
            )
        else:
            prob_stockout = self.probability_stockout_no_rotation(
                n_units, days_left, unit_sales_per_day, lead_time, codelife, verbose
            )

        return prob_stockout

    def probability_stockout_rotation(
        self,
        n_units: np.ndarray,
        days_left: np.ndarray,
        unit_sales_per_day: float,
        lead_time: int,
        codelife: int,
        verbose: int = 0,
    ) -> float:
        """
        Calculates the probability of a stockout occuring at lead_time + 1 days into the future.
        This means stock ordered today will reduce this stockout probability.
            eg. if lead_time = 2, then we'd need to wait 2 days while the stock is in transit. It would be active on day 3. We could consider the probability of a stockout occuring on the third day.

            Consider the case where we have stock with days_left = 3. On day 3 this should also be active, and even though it can't be used on day 4 we shouldn't consider it a stockout if any of the stock remains on day 3.

        To compute this we perform the following steps:
            1. Determine the number of distinct intervals there are seperated by moments that stock becomes inactive, or stock becomes available in store as a result of a delivery. This gives you a series of intervals with a given number of days seperating them.
            2. For the first interval, we start with a guaranteed probability of having the starting number of units in each group. This can be represented as sparse probability distribution which is equal to zero everywhere except at the starting number of units. Populate these distributions for each group.
                2a. Still in the first interval, calculate the mean number of sales as the days in that interval * sales per day.
                2b. Calculate the resulting probability distribution for sales up to 5 standard deviations above the mean using the poisson distribution.
                2c. Starting with the first group, calculate the new distribution of units remaining and the new distribution of sales.
                2d. Take the new distribution of sales and apply this to the next group, and repeat until you reach the final group.
            3. For the second interval, we start with a the probability distribution that we calculated from the first interval. We now begin at the second group, and repeat the process from 2c onwards.
                3a. The mean number of sales may be different if the number of days in the interval has also changed. We will need to recalculate this.
                3b. Propagate the probability distribution forwards as before.
            4. Repeat for all intervals. If you have a delivery on a given day, then you need to update add that to the processed groups for the next interval.

        At the end of this process, we will have a probability distribution for the number of units remaining in each group at the end of the lead time + 1 days. The probability of a stockout is the probability that the active groups on the final day have zero units remaining.

        We will define active groups as those where:
            not_expired: days_left > 0
            entered_stockpile: days_left <= codelife

        To check for active groups after lead_time + 1 days, we need to bear in mind how days are indexed.
        Here the simulation starts at day 0, and hence the lead_time + 1 day is indexed as lead_time.
        Therefore when we check for active groups on the lead_time + 1 day, we need to check for groups that are

        Note to those with above average levels of self-belief. I spent too long getting this logic to fit. I'm pretty sure it's correct even if I couldn't re-explain it to you now. The code below isn't the easiest to follow, but it is to the best of my ability a faithful implementation of the logic above. Change it at your peril, and be warned that even the smallest changes can have outsized impacts on the results.
        """

        max_days_ahead = lead_time + 1

        # Check for active stock at the end of the time period.
        # If not, return a stockout probability of 1.
        active_groups = (days_left - (max_days_ahead - 1) > 0) & (
            days_left - (max_days_ahead - 1) <= codelife
        )
        if not np.any(active_groups):
            if verbose >= 2:
                print("No active groups at end of lead time")
            return 1

        # Initialise the probability distributions for each group.
        p_units_list = []

        for i in range(len(n_units)):
            p_units_list.append(np.zeros(n_units[i] + 1))
            p_units_list[-1][-1] = 1

        # Calculate the intervals where stock expires or enters the system.
        interval_edges = np.sort(
            np.unique(
                np.concatenate(
                    [
                        [0],
                        [max_days_ahead],
                        days_left[days_left <= codelife],
                        days_left[days_left > codelife] - codelife,
                    ]
                )
            )
        )
        interval_edges = interval_edges[interval_edges <= max_days_ahead]
        intervals = np.diff(interval_edges)

        assert np.sum(intervals) == max_days_ahead

        if verbose >= 3:
            print("interval_edges", interval_edges)
            print("intervals", intervals)

        # Start the main logic loop
        days_ahead = 0
        for i, interval in enumerate(intervals):
            # Active groups are those where they haven't expired (days_left > days_ahead), and they have entered the active stockpile (days_left - days_ahead <= codelife)
            not_expired = days_left - days_ahead > 0
            entered_stockpile = days_left - days_ahead <= codelife
            active_groups = not_expired & entered_stockpile
            active_idx = np.argwhere(active_groups).flatten()
            active_p_units_list = [p_units_list[i] for i in active_idx]
            expected_sales = unit_sales_per_day * interval
            max_sales = int(expected_sales + 10 * np.sqrt(expected_sales))

            # Calculate the probability of selling a given number of units
            p_sales = poisson(np.arange(max_sales), expected_sales)
            p_sales = p_sales / np.sum(p_sales)

            if verbose >= 3:
                print("Days Ahead:              ", days_ahead)
                print(
                    "Expired Groups:          ",
                    np.argwhere(np.logical_not(not_expired)).flatten(),
                )
                print("ACTIVE Groups:           ", active_idx)
                print(
                    "Not Entered Stockpile:   ",
                    np.argwhere(np.logical_not(entered_stockpile)).flatten(),
                )
                print("Active Probability Dists:")
                for active_id, p_units in zip(active_idx, active_p_units_list):
                    print("Group", active_id, "p_units", p_units)
                print("Expected Sales:          ", expected_sales)
                print("Probability Sales:       ", p_sales)
                print()

            # Propagate the probability distribution forwards
            active_p_units_list = self.propagate_probability(
                p_sales, active_p_units_list
            )

            # Update the probability distributions for the active groups
            for active_id, all_id in enumerate(active_idx):
                p_units_list[all_id] = active_p_units_list[active_id]

            # Update the days ahead
            days_ahead += interval

        # Calculate the probability of a stockout
        prob_stockout = 1
        for active_id in active_idx:
            prob_stockout *= np.sum(p_units_list[active_id][0])

        return prob_stockout

    def update_units_sales(
        self, p_sales: np.ndarray, p_units: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Given the probability of observing a given number of sales and the probability of having a given number of units, calculate the probability of having a given number of units left, and the probability of observing a given number of sales after selling through the available units.

        In the future we can consider including logsumexp to avoid underflow errors. But for now we'll implement the logic without.
        """
        n_sales, n_units = p_sales.shape[0], p_units.shape[0]
        updated_p_sales = np.zeros(n_sales)
        updated_p_units = np.zeros(n_units)

        # Calculate the probability of selling a given number of units
        p_matix = p_sales[:, np.newaxis] * p_units[np.newaxis, :]

        for i in range(1, n_sales):
            # updated_p_sales[i] = np.sum(np.diagonal(p_matix, offset=-i))
            updated_p_sales[i] = np.trace(p_matix, offset=-i)
        updated_p_sales[0] = 1 - np.sum(updated_p_sales[1:])

        for i in range(1, n_units):
            # updated_p_units[i] = np.sum(np.diagonal(p_matix, offset=i))
            updated_p_units[i] = np.trace(p_matix, offset=i)
        updated_p_units[0] = 1 - np.sum(updated_p_units[1:])
        # TODO: this isn't always correct with large numerical overflows.

        return updated_p_sales, updated_p_units

    def propagate_probability(
        self, p_sales: np.ndarray, p_units_list: list[np.ndarray]
    ) -> list[np.ndarray]:
        updated_p_units_list = []

        for p_units in p_units_list:
            # As we propagate forwards, we update the sales probability distribution and the units probability distribution.
            # When we've updated all the probability distributions, we return the list.
            p_sales, updated_p_units = self.update_units_sales(p_sales, p_units)
            updated_p_units_list.append(updated_p_units)

        return updated_p_units_list

    def probability_stockout_no_rotation(
        self,
        n_units: np.ndarray,
        days_left: np.ndarray,
        unit_sales_per_day: float,
        lead_time: int,
        codelife: int,
        verbose: int = 0,
        account_for_variance: bool = False,
    ) -> float:
        """
        When we don't rotate the stock, we can calculate the probability of a stockout as follows:
            1. Assume the sales for a given day are distributed evenly across all products and are weighted by the unit days available as a fraction of the total.
            2. Calculate the expected sales for each group as (sales per day * remaining days) * (n_units * remaining days for this group) / (sum of n_units * remaining days for all groups).
        This method alone doesn't fully capture the variance. There are methods to do this analytically, but they are more complicated than I want to go into at the moment. By capturing the expected sales as below, we artificially reduce the variance. Instead we can attempt the following.
            3. Calculate the expected number of sales above the available units.
            4. Each sale will be added to all the other sales, so weight the additional sales by the number of groups / (number of groups - 1).
            5. Add these additional sales to the total sales and recalculate the expected sales for each group.

        STOCKOUT PROBABILITY:
            Then the probability of a stockout is the probability of selling at least the n_units for the groups that are active on the final day, which we can calculate from the poisson distribution.
                P(stockout) = 1 - P(selling < n_units) = 1 - sum(P(selling = i) for i in range(n_units))
            And multiply the probabilities together for each group.
        """
        max_days_ahead = lead_time + 1

        # Check for active stock at the end of the time period.
        # If not, return a stockout probability of 1.
        final_active_groups = (days_left - (max_days_ahead - 1) > 0) & (
            days_left - (max_days_ahead - 1) <= codelife
        )
        if not np.any(final_active_groups):
            if verbose >= 2:
                print("No active groups at end of lead time")
            return 1

        # Add an axis to days_ahead, days_left, & n_units so we can broadcast them together
        days_ahead = np.arange(max_days_ahead)[np.newaxis, :]
        days_left = days_left[:, np.newaxis]
        n_units = n_units[:, np.newaxis]

        # Calculate the days that each group is active for, and the corresponding number of units active for those days
        active_groups = (days_left - days_ahead > 0) & (
            days_left - days_ahead <= codelife
        )
        active_unit_days = np.sum(active_groups * n_units, axis=1)

        # Expected sales are just weighted by the number of unit days available
        n_units = n_units.flatten()
        total_sales = unit_sales_per_day * (max_days_ahead)
        expected_sales = total_sales * active_unit_days / np.sum(active_unit_days)

        if account_for_variance:
            extra_sales = 0
            for n_unit, expected_sale in zip(n_units, expected_sales):
                sales = np.arange(int(expected_sale + 5 * np.sqrt(expected_sale)))
                p_sales = poisson(sales, expected_sale)
                extra_sales += np.sum(p_sales[: n_unit + 1] * sales[: n_unit + 1])

            extra_sales = extra_sales * len(n_units) / (len(n_units) - 1)
            total_sales += extra_sales
            expected_sales = total_sales * active_unit_days / np.sum(active_unit_days)

        active_idx = np.argwhere(final_active_groups).flatten()
        prob_stockout = np.zeros(len(active_idx))
        for i, idx in enumerate(active_idx):
            prob_stockout[i] = 1 - np.sum(
                poisson(np.arange(n_units[idx]), expected_sales[idx])
            )

        return np.prod(prob_stockout)

    def simulate_sales(
        self, n_units: np.ndarray, rotation_weighting: bool = True, verbose: int = 0
    ) -> tuple[np.ndarray, int]:
        """Calculate the expected sales for each day by sampling from the poisson distribution with mean equal to the expected sales per day.

        If rotation_weighting is True:
            1. Assume the sales for a given day fall entirely on group with lowest codelife_remaining.
            2. Calculate the actual sales for the first group as the min(sales_today, units remaining).
            3. Calculate the actual sales for the second group as the min(remaining_sales_today, units_remaining).
            4. Repeat for all groups.
            5. For the final group, calculate the actual sales as sales per day * remaining days - expected sales for previous groups.

        If rotation_weighting is False:
            1. Randomly assign the sales for a given day to each group, weighting by the units remaining in that group
        """

        if len(n_units) == 0 or np.sum(n_units) == 0:
            if verbose >= 2:
                print("Sales today: nothing to sell")
            return n_units, 0

        sales_today = ps.rvs(self.unit_sales_per_day)
        original_sales_today = sales_today
        sales = np.zeros(len(n_units), dtype=int)

        if verbose >= 2:
            print("Sales today:", sales_today)

        if rotation_weighting:
            sales[0] = min(sales_today, n_units[0])
            for i in range(1, len(n_units)):
                sales[i] = min(sales_today - sum(sales[:i]), n_units[i])
        else:
            # Sample from the multinomial distribution to get the number of units sold from each group. Never sample more than the minimum number of units in any group.
            # Repeat until we have assigned all the sales for today, or until no units are left.
            while sales_today > 0 and np.sum(n_units - sales) > 0:
                sales_today -= 1
                weights = (n_units - sales) / np.sum(n_units - sales)
                sales += np.minimum(np.random.multinomial(1, weights), n_units)
                if verbose >= 2:
                    print(sales_today, sales, n_units)

            assert np.sum(sales) == np.min([original_sales_today, np.sum(n_units)])

        return n_units - sales, original_sales_today

    def calculate_min_loss_and_variance(
        self, total_days: int, rotation: bool = True
    ) -> tuple[float, float]:
        """
        Calculate the minimum loss and the variance of the loss for a given set of parameters.

        Much of the variance comes from the initial identification of the minimum loss threshold, even if we can accurately quantify the loss once this is better understood.

        NB: this isn't calculating the true variance at the moment, but we hit annoying numerical issues if we do that.
        """

        # In the spirit of total_days being the total number of simulation days available overall, we allocate half the days for finding the minimum threshold, and the second half for calculating the variance.

        total_days = total_days // 2

        min_loss = self.calculate_min_loss(total_days, rotation)
        min_loss, min_loss_std = self.calculate_min_loss_std(
            total_days, stockout_threshold=min_loss.min_stockout_threshold
        )

        return min_loss, min_loss_std
