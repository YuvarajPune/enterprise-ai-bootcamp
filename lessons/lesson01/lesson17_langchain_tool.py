from langchain_core.tools import tool


@tool
def calculate_total(price: float, quantity: int) -> float:
    """Calculate total price from price and quantity."""
    return price * quantity


print(calculate_total.invoke({
    "price": 100,
    "quantity": 3
}))