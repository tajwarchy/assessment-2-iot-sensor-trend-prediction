# IoT Sensor Data Trend Prediction

## Dataset & Industrial Context
Source: Intel Berkeley Research Lab Sensor Network Data
(https://db.csail.mit.edu/labdata/labdata.html)

54 Mica2Dot wireless sensor motes were deployed throughout a research lab
to continuously monitor environmental conditions — temperature, humidity,
light, and battery voltage — every ~31 seconds over 36 days. This
represents a real industrial/field IoT deployment pattern used in smart
buildings, precision agriculture, and environmental/aquaculture
monitoring networks, complete with the real-world noise those systems
produce: dropped transmissions, missing timestamps, and faulty spikes
from low-voltage sensor malfunction.

**Target variable:** future temperature readings for the most complete
sensor mote in the network (trend prediction).

## Pipeline Phases
1. Data acquisition & exploration
2. Data cleaning (dropouts, missing timestamps, noise, spikes)
3. Feature engineering (lags, rolling windows, temporal split)
4. Model training (time-series cross-validated regression)
5. Evaluation & visualization
6. Final documentation & demo video