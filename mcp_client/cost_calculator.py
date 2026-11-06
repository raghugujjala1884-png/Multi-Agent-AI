class CostCalculator:

    def __init__(
        self,
        input_price_per_1m: float,
        output_price_per_1m: float
    ):
        self.input_price_per_1m = input_price_per_1m
        self.output_price_per_1m = output_price_per_1m

    def calculate(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> dict:

        input_cost = (
            input_tokens / 1_000_000
        ) * self.input_price_per_1m

        output_cost = (
            output_tokens / 1_000_000
        ) * self.output_price_per_1m

        total_cost = input_cost + output_cost

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }