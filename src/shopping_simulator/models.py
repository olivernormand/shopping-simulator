from pydantic import Field, BaseModel


class MinimumLoss(BaseModel):
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
