# PID Auto-Tuning Feature

## Overview

The auto-tuning feature uses the **Åström-Hägglund relay feedback method** to automatically determine optimal PID gains for the Stewart platform controller. This method is safer and more reliable than traditional Ziegler-Nichols step response testing.

## Why Åström-Hägglund?

- **Safer**: Uses controlled oscillations instead of pushing the system to instability
- **More accurate**: Directly measures system characteristics at operating conditions
- **Real-time friendly**: Works with noisy sensor data and non-ideal systems
- **Proven**: Widely used in industrial process control

## How It Works

1. **Relay Feedback**: Applies bang-bang control (±amplitude) based on error sign
2. **Oscillation Detection**: Monitors error signal for sustained periodic oscillations
3. **Parameter Extraction**: Measures oscillation amplitude (a) and period (Tu)
4. **Ultimate Gain Calculation**: Ku = 4d/(πa) where d is relay amplitude
5. **PID Tuning**: Applies Ziegler-Nichols rules:
   - Kp = 0.6 × Ku
   - Ki = 1.2 × Ku / Tu
   - Kd = 0.075 × Ku × Tu

## Usage

### Basic Auto-Tune (Both Axes)

```bash
python PID_3d.py --auto-tune
```

This will tune both X and Y axes sequentially and provide averaged results.

### Single Axis Auto-Tune

```bash
# Tune only X-axis
python PID_3d.py --auto-tune --axis x

# Tune only Y-axis
python PID_3d.py --auto-tune --axis y
```

### With Camera Calibration

```bash
python PID_3d.py cal/camera_calib.npz --auto-tune
```

## Step-by-Step Process

1. **Preparation**:
   - Ensure Arduino is connected and servos are working
   - Place ball near the center of the platform
   - Make sure camera has clear view of the ball

2. **Running Auto-Tune**:
   ```bash
   python PID_3d.py --auto-tune
   ```

3. **What to Expect**:
   - Preview window will show "Place ball near center"
   - Press any key when ready (or 'q' to cancel)
   - System will apply relay control and induce oscillations
   - Ball will oscillate back and forth around center
   - Process takes 15-30 seconds per axis
   - Progress displayed: cycle count, error, output

4. **Results**:
   - Terminal displays:
     - Ultimate gain (Ku) and period (Tu)
     - Recommended PID gains (Kp, Ki, Kd)
     - Recommended tilt_gain
   - Plot saved as `autotune_results.png` showing:
     - Error signal over time
     - Relay output over time
     - Zero crossing markers

5. **Applying Results**:
   - Copy recommended gains to your code:
     ```python
     self.Kp = 0.xxx  # From auto-tune output
     self.Ki = 0.xxx
     self.Kd = 0.xxx
     self.tilt_gain = 0.xx
     ```
   - Or save to config file

## Tuning Parameters

You can adjust auto-tuning behavior in `RelayAutoTuner.__init__()`:

- `relay_amplitude`: Size of control output (default: 3.0)
  - Increase for sluggish systems
  - Decrease for very responsive systems
  
- `min_cycles`: Number of oscillation cycles to observe (default: 4)
  - More cycles = more accurate but slower
  
- `timeout`: Maximum time to wait (default: 30.0 seconds)
  
- `convergence_threshold`: Period stability threshold (default: 0.15)
  - Lower = stricter convergence requirement

## Interpreting Results

### Good Auto-Tune
- Oscillations converge to steady amplitude and period
- 4+ complete cycles observed
- Low period variation (<15%)
- Plot shows clear sinusoidal error signal

### Poor Auto-Tune
- Ball falls off platform → reduce relay_amplitude
- No oscillations detected → increase relay_amplitude or tilt_gain
- Erratic oscillations → check mechanical issues, reduce friction
- Timeout reached → system may be too slow or relay_amplitude too small

## Fine-Tuning Recommendations

Auto-tuned gains are **starting points**. Adjust based on performance:

- **Too much overshoot**: Reduce Ki by 20-50%
- **Oscillations persist**: Increase Kd by 20-30%
- **Too slow response**: Increase Kp by 10-20%
- **Platform tilts too much**: Reduce tilt_gain
- **Not enough tilt**: Increase tilt_gain

## Troubleshooting

### "Ball not detected"
- Ensure ball is visible in camera view
- Check HSV color thresholds in config.json
- Verify camera calibration if using

### "Insufficient oscillation data"
- Ball may have fallen off → reduce relay_amplitude
- Increase timeout for slower systems
- Check that servos are responding to commands

### "Failed to compute oscillation periods"
- Oscillations not sustained → mechanical issues?
- Try different relay_amplitude (±1.0)
- Verify ball tracking is stable

### Unstable Results Between Runs
- Normal variation ±10-20%
- Average multiple runs for best results
- Ensure consistent starting conditions

## Advanced: Manual Relay Tuning

If auto-tune struggles, manually adjust `RelayAutoTuner` parameters:

```python
tuner = RelayAutoTuner(
    relay_amplitude=2.5,  # Start lower for sensitive systems
    min_cycles=6,         # More cycles for accuracy
    timeout=45.0,         # More time for slow systems
    convergence_threshold=0.10  # Stricter convergence
)
```

## Theory Reference

**Relay Feedback Method** (Åström & Hägglund, 1984):
- Describes a limit cycle in the system
- Ultimate gain: Ku = 4d/(πa)
- Oscillation period = Ultimate period (Tu)

**Ziegler-Nichols PID Rules**:
- Based on ultimate gain and period
- Provides 1/4 decay ratio response
- Conservative tuning (reduces overshoot)

## Example Output

```
======================================================================
AUTO-TUNE RESULTS
======================================================================
✓ Auto-tuning successful: Ku=12.45, Tu=0.847s

System Characteristics:
  Ultimate Gain (Ku):   12.450
  Ultimate Period (Tu): 0.847 seconds

Recommended PID Gains (Ziegler-Nichols PID):
  Kp (Proportional): 7.470
  Ki (Integral):     17.661
  Kd (Derivative):   0.790

Recommended Control Parameters:
  Tilt Gain:         0.50

To use these gains, update your config.json or modify the code:
  self.Kp = 7.470
  self.Ki = 17.661
  self.Kd = 0.790
  self.tilt_gain = 0.50

NOTE: These are starting values. Fine-tune based on performance.
======================================================================
```

## Safety Notes

⚠️ **Important Safety Considerations**:
- Auto-tune induces oscillations → secure the platform
- Ball may fall off during tuning (normal, restart if needed)
- Start with conservative relay_amplitude (2.0-3.0)
- Monitor first run closely
- Press 'q' anytime to abort

## References

- Åström, K. J., & Hägglund, T. (1984). "Automatic Tuning of Simple Regulators with Specifications on Phase and Amplitude Margins"
- Ziegler, J. G., & Nichols, N. B. (1942). "Optimum Settings for Automatic Controllers"
