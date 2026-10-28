# Data removed

The VTK weather data has been removed because it is the intellectual property (IP) of Caterpillar Inc.

This directory intentionally contains only this notice. Supply your own authorized data locally to run the project.

## Sample VTK Format

The excerpt below shows the header and two example point rows.

```text
DATASET STRUCTURED_POINTS
DIMENSIONS 1 1 1
ORIGIN 0 0 0
SPACING 1 1 1
POINTS 1000 float

21.123 24.123 2.123 16.123
21.123 24.123 2.123 543
```

**Point row layout:** `X Y Z Intensity`

| Column | Description |
| --- | --- |
| `X` | Position along the X axis |
| `Y` | Position along the Y axis |
| `Z` | Position along the Z axis |
| `Intensity` | LiDAR return intensity |

