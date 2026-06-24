# cogen-est-baseline

Companion repository to **Cogen-Estimation**. Reproduces the client's Excel-based P50
conditional-lookup methodology as a formal, testable baseline for Spanish electricity
balancing market indicator forecasting.

## Goal

Given a long-horizon (18+ month) day-ahead price forecast (indicator 600), estimate
the following ESIOS balancing market indicators at quarter-hourly resolution:

| ID    | Name                                              | Family |
|-------|---------------------------------------------------|--------|
| 634   | Precio reserva regulación secundaria a bajar      | aFRR   |
| 682   | Precio energía regulación secundaria a subir      | aFRR   |
| 683   | Precio energía regulación secundaria a bajar      | aFRR   |
| 708   | Precio medio restricciones técnicas diario F-II   | TC     |
| 2197  | Precio energías balance mFRR activación programada| mFRR   |
| 10250 | Volumen neto asignación energías mFRR             | mFRR   |

## Approach

The baseline method (from the client's `Genera_precios_p50.xlsm`) works as follows:

1. Discretise the day-ahead price (PMD / indicator 600) into €5/MWh bands.
2. For each target indicator, compute the historical P50 (median) of the spread or
   value conditional on `(hour, PMD-band)`.
3. At inference, look up the appropriate P50 value for each quarter-hourly period
   using the forecasted PMD's band and hour.

This repository implements, validates, and benchmarks that baseline so that any
subsequent ML approach can be compared against it on equal footing.

## Data conventions

- **634**: used from 20/11/2024 onwards (post regime split with 2130).
- **2197**: backfilled before 10/12/2024 using 676/677, with the sign of 10250
  determining which indicator applies at each timestep.
- **10250**: always at 15-min frequency.
- **estimated_600**: client-provided hourly forecast, expanded to 15-min by
  forward-filling the hourly value across its four quarter-hourly slots.

## Setup

```bash
uv sync
cp .env.example .env  # fill in ESIOS_API_TOKEN
```

## Running tests

```bash
uv run pytest
```

## Running the baseline evaluation

```bash
# From a local directory with id_*.csv files
uv run python -m cogen_est_baseline.pipelines.baseline_pipeline --local-path /path/to/csvs/

# From a ClearML Dataset by ID
uv run python -m cogen_est_baseline.pipelines.baseline_pipeline --dataset-id <dataset-id>

# Custom train/test split date
uv run python -m cogen_est_baseline.pipelines.baseline_pipeline --dataset-id <id> --split-date 2025-01-01
```
