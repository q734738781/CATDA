import logging
from typing import Type, Dict, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

# Pint import for unit conversion
try:
    import pint
    from pint import UnitRegistry, DimensionalityError, UndefinedUnitError
    PINT_AVAILABLE = True
    ureg = UnitRegistry() # Create a global unit registry
except ImportError:
    PINT_AVAILABLE = False
    logging.warning("pint package not found. UnitConverterTool will not function. Run `pip install pint`")
    # Define dummy classes for type hinting if pint is not available
    class UnitRegistry: pass
    class DimensionalityError(Exception): pass
    class UndefinedUnitError(Exception): pass
    ureg = None

logger = logging.getLogger(__name__)

# --- Tool Input Schema ---
class UnitConverterInput(BaseModel):
    """Input schema for the UnitConverterTool."""
    value: float = Field(description="The numerical value to convert.")
    from_unit: str = Field(description="The original unit symbol or name (e.g., 'degC', 'bar', 'psi') compatible with the 'pint' library.")
    to_unit: str = Field(description="The target unit symbol or name (e.g., 'K', 'atm', 'degF') compatible with the 'pint' library.")

class UnitConverterTool(BaseTool):
    """
    Converts a numerical value from a specified source unit to a specified target unit.
    Uses the 'pint' library for handling unit conversions, which can often parse
    variations in unit names (e.g., 'degC', 'degree Celsius').
    """
    name: str = "UnitConverter"
    description: str = (
        "Convert a numerical value from a source unit to a target unit (e.g., convert '25 degC' to 'K'). "
        "Uses the `pint` library. Input requires value, from_unit, and to_unit."
    )
    args_schema: Type[BaseModel] = UnitConverterInput

    # No suffix map needed anymore

    def __init__(self, **kwargs):
        """Initialize the tool."""
        super().__init__(**kwargs)
        if not PINT_AVAILABLE:
            raise ImportError("Pint library is not available. Please install it using `pip install pint`.")
        logger.info("UnitConverterTool initialized.")

    def _run(self, value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Perform the unit conversion using pint."""
        logger.info(f"Attempting conversion: {value} {from_unit} to {to_unit}")

        if not PINT_AVAILABLE:
            return {"error": "Pint library not available for unit conversion."}
        if not ureg:
             return {"error": "Pint Unit Registry not initialized."}

        # Pint's parser is generally flexible with unit string variations (e.g., degC, °C, degree_celsius)
        # No explicit normalization needed here unless specific issues arise.
        from_unit_str = from_unit

        try:
            # Create quantity with the source unit
            quantity = ureg.Quantity(value, from_unit_str)
            logger.debug(f"Created quantity: {quantity}")

            # Perform conversion to the target unit
            converted_quantity = quantity.to(to_unit)
            logger.info(f"Conversion successful: {quantity} -> {converted_quantity}")

            return {
                "original_value": value,
                "original_unit": from_unit_str,
                "target_unit": to_unit,
                "converted_value": converted_quantity.magnitude,
                "result_string": f"{converted_quantity:.4f~P}" # Pretty formatting
            }

        except UndefinedUnitError as e:
             logger.error(f"Unit conversion error: Undefined unit - {e}")
             # Include the problematic unit string in the error message
             if str(e).startswith("'" + from_unit_str):
                 error_msg = f"Invalid or undefined source unit specified: '{from_unit_str}'"
             elif str(e).startswith("'" + to_unit):
                  error_msg = f"Invalid or undefined target unit specified: '{to_unit}'"
             else:
                 error_msg = f"Invalid or undefined unit specified: {e}"
             return {"error": error_msg}
        except DimensionalityError as e:
            logger.error(f"Unit conversion error: Dimensionality mismatch - {e}")
            return {"error": f"Cannot convert between incompatible units: {from_unit_str} and {to_unit}. Details: {e}"}
        except Exception as e:
            logger.error(f"An unexpected error occurred during unit conversion: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred during conversion: {e}"}

    async def _arun(self, value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Asynchronous execution (calls synchronous version)."""
        # Consider implementing true async if pint supports it or if needed for performance
        # logger.warning("_arun (async unit converter) is not implemented. Falling back to sync.")
        # For now, just call the synchronous version
        return self._run(value, from_unit, to_unit)

# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    if not PINT_AVAILABLE:
        logger.info("Pint library not installed. Cannot run UnitConverterTool tests.")
    else:
        logger.info("Testing UnitConverterTool...")
        tool = UnitConverterTool()

        # Test 1: Temperature C to K
        test_input_1 = {"value": 25.0, "from_unit": "degC", "to_unit": "K"}
        logger.info(f"Input 1: {test_input_1}")
        result1 = tool.run(test_input_1)
        logger.info(f"Result 1: {result1}")
        # Expected: roughly 298.15 K

        # Test 2: Pressure bar to psi
        test_input_2 = {"value": 1.5, "from_unit": "bar", "to_unit": "psi"}
        logger.info(f"Input 2: {test_input_2}")
        result2 = tool.run(test_input_2)
        logger.info(f"Result 2: {result2}")
        # Expected: roughly 21.75 psi

        # Test 3: Invalid source unit
        test_input_3 = {"value": 100.0, "from_unit": "unknown_unit", "to_unit": "K"}
        logger.info(f"Input 3: {test_input_3}")
        result3 = tool.run(test_input_3)
        logger.info(f"Result 3: {result3}")
        # Expected: Error about undefined unit 'unknown_unit'

        # Test 4: Incompatible units
        test_input_4 = {"value": 50.0, "from_unit": "degC", "to_unit": "kg"}
        logger.info(f"Input 4: {test_input_4}")
        result4 = tool.run(test_input_4)
        logger.info(f"Result 4: {result4}")
        # Expected: Error about dimensionality mismatch

        # Test 5: Percentage to dimensionless
        test_input_5 = {"value": 85.0, "from_unit": "percent", "to_unit": "dimensionless"}
        logger.info(f"Input 5: {test_input_5}")
        result5 = tool.run(test_input_5)
        logger.info(f"Result 5: {result5}")
        # Expected: 0.85 dimensionless

        # Test 6: Using variation (degree Celsius)
        test_input_6 = {"value": 0.0, "from_unit": "degree Celsius", "to_unit": "degF"}
        logger.info(f"Input 6: {test_input_6}")
        result6 = tool.run(test_input_6)
        logger.info(f"Result 6: {result6}")
        # Expected: 32 degF 