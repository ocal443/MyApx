from __future__ import annotations

import ctypes
import time
from fractions import Fraction
from typing import Callable, Tuple

import serial

from festo_ctrl.can_objects import *

AXIS_FEED_MM_PER_REV = 10.0
MOTOR_TO_AXIS_GEAR_RATION = 1.0


class FestoDevice:
    def __init__(self, port: str):
        self.serial = serial.Serial(
            port=port,
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=None,
            xonxoff=False,
            rtscts=True,
            dsrdtr=False,
        )
        self.last_object_read: Tuple[int, int, int] = (0x0, 0x0, 0x0)

    def _serial_write(self, msg: str) -> None:
        self.serial.write((msg + "\r").encode())

    def _serial_read(self) -> str:
        return self.serial.read_until(b"\r")[:-1].decode()

    def _sdo_access(self, obj: CanObjectBase, wval: int | None) -> int:
        if wval is None:
            rq = f"?{obj.INDEX:04x}{obj.SUB_INDEX:02x}"
        else:
            rq = f"={obj.INDEX:04x}{obj.SUB_INDEX:02x}:{wval:0{obj.SIZE * 2}x}"

        self._serial_write(rq)
        rs = self._serial_read()

        if rs.startswith("!") and not rs.endswith("!"):
            raise RuntimeError(SdoAccessError(int(rs[1:9], 16)))
        if not rs.startswith("="):
            raise RuntimeError(f"Failed SDO access: rq:'{rq}', rs:'{rs}'")

        rs_idx, rs_sub_idx, rs_val = (
            int(rs[1:5], 16),
            int(rs[5:7], 16),
            int(rs[8:], 16),
        )
        if (rs_idx, rs_sub_idx) != (obj.INDEX, obj.SUB_INDEX):
            raise RuntimeError(f"Invalid SDO response: rq:'{rq}', rs:'{rs}'")

        if self.last_object_read != (obj.INDEX, obj.SUB_INDEX, rs_val):
            self.last_object_read = (obj.INDEX, obj.SUB_INDEX, rs_val)

        return rs_val

    def object_read(self, obj: CanObjectBase) -> int:
        return self._sdo_access(obj, None)

    def object_write(self, obj: CanObjectBase, value: int) -> int:
        return self._sdo_access(obj, value)


class SerialController:
    def __init__(self, device: FestoDevice):
        self.device = device

    def do_homing(self) -> Callable:
        self.device._serial_write("OW:1:0010:00000022")
        res = self.device._serial_read()
        assert res == "OK!"

        start = time.time()

        def is_done():
            nonlocal start
            while time.time() - start < 10:
                time.sleep(0.1)
            return True

        return is_done

    def do_program(self, program_id: int) -> Callable:
        self.device._serial_write(f"OW:1:0010:{program_id:04x}0021")
        res = self.device._serial_read()
        assert res == "OK!"

        start = time.time()

        def is_done():
            nonlocal start
            while time.time() - start < 10:
                time.sleep(0.1)
            return True

        return is_done


class CANController:
    def __init__(self, device: FestoDevice):
        self.device = device

        self.control = Control(self.device)
        self.status = Status(self.device)
        self.enable_logic = EnableLogic(self.device)
        self.modes_of_operation = ModesOfOperation(self.device)
        self.modes_of_operation_display = ModesOfOperationDisplay(self.device)
        self.limit_switch_polarity = LimitSwitchPolarity(self.device)
        self.target_position = TargetPosition(self.device)
        self.profile_velocity = ProfileVelocity(self.device)
        self.profile_acceleration = ProfileAcceleration(self.device)
        self.profile_deceleration = ProfileDeceleration(self.device)
        self.manufacturer_status_word1 = ManufacturerStatusWord1(self.device)
        self.position_factor_numerator = PositionFactorNumerator(self.device)
        self.position_factor_divisor = PositionFactorDivisor(self.device)

    def _set_units(self) -> None:
        motor_to_gear = Fraction(MOTOR_TO_AXIS_GEAR_RATION)
        ticks_per_rev = Fraction(65536)
        axis_feed = Fraction(AXIS_FEED_MM_PER_REV)
        fraction = (motor_to_gear * ticks_per_rev) / axis_feed
        num, denom = int(fraction.numerator), int(fraction.denominator)
        self.position_factor_numerator.write(num)
        self.position_factor_divisor.write(denom)

    def move_to(self, pos: float, v: float = 1, a: float = 0.1, relative=False):
        self._set_units()
        to_units = lambda x: int(x)  # int(x * 10_000)
        pos, v, a = to_units(pos), to_units(v), to_units(a)
        target_mode = 1  # profile position mode
        current_mode = self.modes_of_operation_display.read()
        start_time = time.time()

        def timed_out():
            return time.time() - start_time > 1

        self.modes_of_operation.update(profile_position_mode=True)
        while current_mode != target_mode and not timed_out():
            time.sleep(0.1)

        current_mode = self.modes_of_operation_display.read()
        assert current_mode == target_mode, f"wrong operation mode: {current_mode}"

        assert v == ctypes.c_uint32(v).value
        assert a == ctypes.c_uint32(a).value
        assert pos == ctypes.c_int32(pos).value

        self.target_position.write(pos)
        self.profile_velocity.write(v)
        self.profile_acceleration.write(a)
        self.profile_deceleration.write(a)

        self.control.update(relative=1 if relative else 0, new_set_point=0)
        self.control.update(new_set_point=1)

        while not self.status.read("set_point_acknowledge"):
            time.sleep(0.1)

        print("Acknowledged")

        while not self.status.read("target_reached"):
            time.sleep(0.1)

        print("Reached")

    def _get_drive_state(self):
        status_value = self.status.read()
        for field_name in [
            "not_ready_to_switch_on",
            "switch_on_disabled",
            "ready_to_switch_on",
            "switched_on",
            "operation_enable",
            "quick_stop_active",
            "fault_reaction_active",
            "fault",
        ]:
            if getattr(Status, field_name).read(status_value):
                return field_name
        raise RuntimeError("Failed to find drive state")

    def _wait_for_drive_state(self, state_name: str, timeout_ms=3000):
        start_time = time.time()
        current_state = self._get_drive_state()
        while time.time() - start_time < timeout_ms / 1000:
            if current_state == state_name:
                return True
            if current_state == "fault":
                raise RuntimeError(
                    "Device entered FAULT state while waiting for {state_name}."
                )
            time.sleep(0.1)
            current_state = self._get_drive_state()
        raise TimeoutError(
            f"Timeout waiting for state {state_name}. Last state: {current_state}"
        )

    @property
    def stopped(self) -> bool:
        return not self.status.read("drive_is_moving")

    def wait_for_stopped(self, timeout=None, poll_interval: float = 0.1) -> bool:
        start_time = time.time()
        while self.status.read("drive_is_moving"):
            if timeout is not None and time.time() - start_time > timeout:
                return False
            time.sleep(poll_interval)
        return True

    def can_start(self):
        current_state = self._get_drive_state()
        print(f"Initial state: {current_state}.")

        # Check if limit switches are configured correctly - avoid hitting endpoints
        if not self.limit_switch_polarity.read("normally_closed"):
            raise RuntimeError("Expect normally closed limit switches.")

        self.enable_logic.write(2)  # physical signal + CAN logic for drive to function
        if not self.status.read("remote"):
            time.sleep(0.2)  # Give it a moment
            if not self.status.read("remote"):
                raise RuntimeError(
                    "Device did not switch to remote mode after enabling logic."
                )
        print("Remote mode enabled.")

        # Refresh state
        current_state = self._get_drive_state()
        print(f"State after enabling remote: {current_state}")

        if current_state == "fault":
            print("Device in FAULT state. Attempting fault reset...")
            self.control.update(reset_fault=1)
            time.sleep(0.1)
            self.control.update(reset_fault=0)
            time.sleep(0.1)
            self._wait_for_drive_state("switch_on_disabled")
            current_state = "switch_on_disabled"
            print(f"Transitioned to: {current_state}")

        if current_state == "switch_on_disabled":
            self.control.update(shutdown=1)
            self._wait_for_drive_state("ready_to_switch_on")
            current_state = "ready_to_switch_on"
            print(f"Transitioned to: {current_state}")

        if current_state == "ready_to_switch_on":
            self.control.update(switch_on=1)
            self._wait_for_drive_state("switched_on")
            current_state = "switched_on"
            print(f"Transitioned to: {current_state}")

        if current_state == "switched_on":
            self.control.update(enable_operation=1)
            self._wait_for_drive_state("operation_enable")
            current_state = "operation_enabled"
            print(f"Transitioned to: {current_state}")

        if current_state != "operation_enabled":
            raise RuntimeError(
                f"Failed to initialize drive. Final state: {current_state}"
            )

        print("Drive initialized successfully to OPERATION_ENABLED.")


class FestoController(CANController, SerialController):
    def __init__(self, port: str):
        device = FestoDevice(port)
        SerialController.__init__(self, device)
        CANController.__init__(self, device)
