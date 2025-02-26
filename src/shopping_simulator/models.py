from pydantic import Field, BaseModel
import numpy as np


class SimulationResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    wasted_units: np.ndarray = Field(
        description="The number of units wasted on each day of the simulation"
    )
    cases_ordered: np.ndarray = Field(
        description="The number of cases ordered on each day of the simulation"
    )
    stockout: np.ndarray = Field(
        description="Whether there was a stockout on each day of the simulation"
    )

    units_sold: np.ndarray = Field(
        description="The number of units sold on each day of the simulation"
    )

    eod_units: np.ndarray = Field(
        description="The number of units left in stock at the end of each day of the simulation"
    )


class MinimumLossResult(BaseModel):
    min_loss: float = Field(
        description="The minimum loss that can be achieved. This equals the waste loss + the availability loss."
    )

    min_waste_loss: float = Field(
        description="The minimum waste loss that can be achieved."
    )

    min_availability_loss: float = Field(
        description="The minimum availability loss that can be achieved."
    )

    min_stockout_threshold: float = Field(
        description="The stockout threshold that achieves the minimum loss. This is the same as the acceptable probability of stockout before ordering more stock."
    )

    mean_units_sold_per_day: float = Field(
        description="The average units sold per day that are sold in the minimum loss scenario."
    )

    mean_eod_units: float = Field(
        description="The average eod units that are left in stock in the minimum loss scenario after the days sales."
    )

    thresholds: list[float] = Field(
        description="The thresholds that were tested to find the minimum loss. These are the stockout thresholds that were tested."
    )

    loss: list[float] = Field(
        description="The loss that was achieved for each threshold. This is the waste loss + the availability loss."
    )

    availability_loss: list[float] = Field(
        description="The availability loss that was achieved for each threshold."
    )

    waste_loss: list[float] = Field(
        description="The waste loss that was achieved for each threshold."
    )

    units_sold: list[float] = Field(
        description="The units sold per day for each threshold."
    )

    eod_units: list[float] = Field(
        description="The eod units that are left in stock after the days sales for each threshold."
    )
