# MTE 380 Stewart Platform Project

A 3-servo Stewart platform with inverse kinematics implementation and interactive 3D visualization.

## Features

- Inverse kinematics calculation for a 3-servo circular Stewart platform
- Interactive 3D visualization using matplotlib
- Real-time control with arrow keys
- Configurable link lengths and platform geometry

## Setup Instructions

### 1. Create Virtual Environment

Run the setup script to create a virtual environment and install dependencies:

```bash
bash setup_env.sh
```

## Running the Visualization

Once the environment is set up and activated:

```bash
source venv/bin/activate
python3 inverseKinematics.py
```

## Arduino Hardware Control (PlatformIO)

The project includes Arduino code to control three servos via an Adafruit PCA9685 PWM/Servo Shield.

### Setup

1. Install [PlatformIO](https://platformio.org/) (VSCode extension recommended)
2. Connect your Arduino with the Adafruit PCA9685 shield
3. Connect three servos to channels 0, 1, and 2 on the shield
4. Use a separate 5-6V power supply for the servos (connect grounds together)

### Build and Upload

```bash
pio run --target upload
```

Or use the PlatformIO IDE buttons in VSCode.

### Behavior

Control the platform using arrow keys, run the arduino controller python file, after uploading the arduino code.

### Configuration

Edit `src/main.cpp` to adjust:
- `SERVO_MIN` / `SERVO_MAX` - Pulse width calibration for your servos
- Servo channels if using different connections
- Motion parameters (amplitude, frequency)

## Sources

- [Inverse Kinematics of a Stewart Platform](https://raw.org/research/inverse-kinematics-of-a-stewart-platform/)
