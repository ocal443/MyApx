# Festo Control Package

A Python package for controlling Festo CMMP-AS servo positioning controllers using both serial and CANopen protocols. This package provides a high-level interface for precise positioning control of linear stages and other motion control applications.

## ⚠️ CRITICAL SAFETY WARNINGS ⚠️

⚠️ **SAFETY WARNINGS** ⚠️
- Velocity/acceleration are in **uncalibrated units** - start with low values (v=1, a=0.1)
- Can drive motors at full torque/speed - incorrect parameters may damage hardware
- Configure limit switches, torque limits, and emergency stops before use
- Test in controlled environment with safety systems verified
- **USE AT YOUR OWN RISK**

## Features

- **Hardware Integration**: Support for Festo CMMP-AS controllers
- **Dual Communication Protocols**: Supports both serial and CANopen communication
- **Precise Positioning**: Position control with configurable velocity and acceleration profiles
- **State Management**: Comprehensive drive state monitoring and control
- **Error Handling**: Robust error detection and recovery mechanisms

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
uv add git+https://dev.azure.com/OttokarCallewaert/Scrap/_git/APX_FestoController
```

## Quick Start

### Basic Setup and Movement

```python
from festo_ctrl import FestoController

# Initialize controller with COM port
controller = FestoController("COM3")  # Use appropriate port for your system

# Initialize the drive to operational state
controller.can_start()

# Move to absolute position (position in mm, velocity/acceleration in uncalibrated units)
controller.move_to(
    pos=100.0,     # Target position in mm
    v=1.0,         # Velocity (UNCALIBRATED UNITS - start low!)
    a=0.1,         # Acceleration (UNCALIBRATED UNITS - start low!)
    relative=False # Absolute positioning
)

# Relative movement
controller.move_to(
    pos=10.0,      # Move 10mm from current position
    v=1.0,         # Velocity (UNCALIBRATED UNITS - start low!)
    a=0.1,         # Acceleration (UNCALIBRATED UNITS - start low!)
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
controller.profile_velocity.write(1)   # Velocity (UNCALIBRATED UNITS!)
controller.profile_acceleration.write(1) # Acceleration (UNCALIBRATED UNITS!)

# Control drive states
controller.control.update(enable_operation=1)
```

## API Reference

### FestoController

The main controller class that provides both serial and CAN communication capabilities.

#### Methods

- `__init__(port: str)`: Initialize controller with specified serial port
- `can_start()`: Initialize drive to operational state with full state machine handling
- `move_to(pos, v=1, a=0.1, relative=False)`: Execute positioning move
- `do_homing() -> Callable`: Start homing operation (returns completion checker)
- `do_program(program_id: int) -> Callable`: Execute stored program

### CAN Objects

The package provides direct access to CANopen objects for advanced control:

- `control`: Drive control word (0x6040)
- `status`: Drive status word (0x6041) 
- `modes_of_operation`: Operation mode selection (0x6060)
- `target_position`: Target position (0x607A)
- `profile_velocity`: Motion velocity (0x6081)
- `profile_acceleration`: Motion acceleration (0x6083)
- `profile_deceleration`: Motion deceleration (0x6084)

Each object supports:
- `read(field=None)`: Read object value or specific field
- `write(value)`: Write complete object value
- `update(**fields)`: Update specific fields

## Project Structure

```
festo-ctrl/
├── src/festo_ctrl/
│   ├── __init__.py              # Package initialization
│   ├── stage_controller.py      # Main controller classes
│   └── can_objects.py           # CANopen object definitions
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```
