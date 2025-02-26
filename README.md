## Shopping Simulator

This package determines the best availability and waste characteristics of a product in a store. This depends on four parameters:

1. `codelife` defines the number of days a product is available in the store once it hits the shelves.
2. `unit_sales_per_day` defines the number of units sold per day.
3. `units_per_case` defines the number of units in a case. The case is the fundamental unit of the product in the retail supply chain.
4. `lead_time` defines the number of days taken for the demand signal to lead to a stock replenishment.

Based on these parameters, `shopping-simulator` is able to determine the best availability and waste performance of a product in the store. 

---
## Basic Usage

The basic usage of the package is as follows:

```python
from shopping_simulator.simulator import LossSimulation

sim = LossSimulation(
    codelife=3, 
    unit_sales_per_day=5, 
    units_per_case=4, 
    lead_time=3
)

min_loss, min_loss_std = sim.calculate_min_loss_and_variance(total_days=20000)

print(min_loss, min_loss_std)
>>> 0.1037, 0.0067
```

This calculates the minimum loss (the sum of waste and unavailability) of the product in the store, along with its standard deviation. The simulation runs for 20,000 days to ensure statistical stability.

For more detailed results, you can use the `calculate_min_loss()` method which returns additional metrics like waste loss, availability loss, and average stock levels:

```python
results = sim.calculate_min_loss(total_days=2000)
print(results.min_loss, results.min_availability_loss, results.min_waste_loss)
>>> 0.1091, 0.0800, 0.0291
```