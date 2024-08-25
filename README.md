## Shopping Simulator

This package determines the best availabiliity and waste characteristics of a product in a store. This depends on four parameters:

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
    unit_sales_per_day=10, 
    units_per_case=4, 
    lead_time=3
)

min_loss, min_loss_threshold, outputs = sim.calculate_min_loss(
    total_days=1000,
    rotation=True
)

print(min_loss)
>>> 0.1194
```
This calculates the minimum loss (the sum of the waste and percentage of time unavailable) of the product in the store. The `rotation` parameter indicates that stock is rotated.