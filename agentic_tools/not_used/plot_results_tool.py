import logging
import os
import uuid
from pathlib import Path
from typing import Type, Dict, Any, List

import pandas as pd
import matplotlib
# Use non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PLOTS_DIR = os.getenv("PLOTS_DIR", os.getcwd())

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

class PlotResultsInput(BaseModel):
    """Input schema for PlotResultsTool."""
    dataframe: List[Dict[str, Any]] = Field(description="Tabular data as list of dicts (each dict = row).")
    plot_args: Dict[str, Any] = Field(description="Matplotlib kwargs. Must include 'kind', 'x', 'y'. Other kwargs optional.")

class PlotResultsTool(BaseTool):
    """Render quick plots from a DataFrame the agent already holds."""

    name: str = "PlotResults"
    description: str = (
        "Render a matplotlib plot from tabular data. Provide DataFrame as list of dicts and plotting args (kind, x, y). "
        "Returns the file path of the saved PNG plot."
    )
    args_schema: Type[BaseModel] = PlotResultsInput

    def _run(self, dataframe: List[Dict[str, Any]], plot_args: Dict[str, Any]) -> str:
        if not dataframe:
            return "Error: dataframe cannot be empty."
        if not plot_args or not isinstance(plot_args, dict):
            return "Error: plot_args must be provided as a dict."
        if "kind" not in plot_args or "x" not in plot_args or "y" not in plot_args:
            return "Error: plot_args must include 'kind', 'x', and 'y'."

        try:
            df = pd.DataFrame(dataframe)
        except Exception as e:
            logger.error(f"Failed to construct DataFrame: {e}")
            return f"Error: could not construct DataFrame: {e}"

        kind = plot_args.pop("kind")
        x = plot_args.pop("x")
        y = plot_args.pop("y")
        try:
            ax = df.plot(kind=kind, x=x, y=y, **plot_args)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            plt.tight_layout()
            filename = f"plot_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(PLOTS_DIR, filename)
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=150)
            plt.close()
            logger.info(f"Plot saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error while creating plot: {e}")
            return f"Error: could not create plot: {e}"

    async def _arun(self, dataframe: List[Dict[str, Any]], plot_args: Dict[str, Any]):
        logger.warning("_arun (async plotting) not implemented, falling back to sync.")
        return self._run(dataframe, plot_args) 