# Festo Control Package

A Python package for controlling Festo CMMP-AS servo positioning controllers using both serial and CANopen protocols. This package provides a high-level interface for precise positioning control of linear stages and other motion control applications.

## Features

- **Dual Communication Protocols**: Supports both serial and CANopen communication
- **DSP402 Compliant**: Implements CANopen DSP402 drive profile for servo control
- **Precise Positioning**: High-accuracy position control with configurable velocity and acceleration profiles
- **State Management**: Comprehensive drive state monitoring and control
- **Error Handling**: Robust error detection and recovery mechanisms
- **Hardware Integration**: Direct support for Festo CMMP-AS controllers

## Hardware Compatibility

This package is designed for use with Festo CMMP-AS servo positioning controllers. For detailed protocol information, refer to the official Festo documentation:
- CANopen manual: P.BE-CMMP-CO-SW "CMMP-AS servo positioning controller"

## Installation

### From Source

```bash
git clone <repository-url>
cd festo-ctrl
pip install -e .
```

### Using uv (recommended)

```bash
uv add festo-ctrl
```

## Requirements

- Python ≥ 3.10
- PySerial ≥ 3.5
- Windows/Linux/macOS with available serial ports

## Quick Start

### Basic Setup and Movement

```python
from festo_ctrl import FestoController

# Initialize controller with COM port
controller = FestoController("COM3")  # Use appropriate port for your system

# Initialize the drive to operational state
controller.can_start()

# Move to absolute position (units in mm)
controller.move_to(
    pos=100.0,     # Target position in mm
    v=50.0,        # Velocity 
    a=10.0,        # Acceleration
    relative=False # Absolute positioning
)

# Relative movement
controller.move_to(
    pos=10.0,      # Move 10mm from current position
    v=25.0,        # Velocity
    a=5.0,         # Acceleration  
    relative=True  # Relative positioning
)
```

### Homing Operation

```python
# Perform homing sequence
wait_for_homing = controller.do_homing()
# Homing operation runs asynchronously - check completion if needed
```

### Advanced Usage - Direct CAN Object Access

```python
# Access individual CAN objects for advanced control
controller = FestoController("COM3")

# Read device status
status = controller.status.read()
is_moving = controller.status.read("drive_is_moving")
target_reached = controller.status.read("target_reached")

# Configure operation mode
controller.modes_of_operation.update(profile_position_mode=True)

# Set motion parameters directly
controller.target_position.write(500)  # Target position
controller.profile_velocity.write(100) # Velocity
controller.profile_acceleration.write(50) # Acceleration

# Control drive states
controller.control.update(enable_operation=1)
```
