"""Utility functions for FLRW-Net."""

def get_triangulation_params(triangulation: str) -> dict[str, int]:
    """Set the parameters that specify the spatial triangulations."""
    if triangulation == "5-cell":
        return {
            "n1": 10,
            "n2": 10,
            "n3": 5,
            "nte": 3,
        }
    elif triangulation == "16-cell":
        return {
            "n1": 24,
            "n2": 32,
            "n3": 16,
            "nte": 4,
        }
    elif triangulation == "600-cell":
        return {
            "n1": 720,
            "n2": 1200,
            "n3": 600,
            "nte": 5,
        }
    msg = f"The given triangulation {triangulation} is invalid. Please choose from '5-cell', '16-cell', '600-cell'."
    raise ValueError(msg)
