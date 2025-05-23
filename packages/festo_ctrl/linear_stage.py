from __future__ import annotations

import ctypes
import enum
import time
from fractions import Fraction
from typing import Callable, Dict, List, Tuple, overload

import serial

from festo_can_objs import CAN_OBJECTS_DESCRIPTIONS, SdoAccessError

AXIS_FEED_MM_PER_REV = 10.0
MOTOR_TO_AXIS_GEAR_RATION = 1.0


# https://www.festo.com/net/en_corp/SupportPortal/Files/777998/CMMP-AS-CO_2007-08_557344g1.pdf
class FestoCanObject:
    def __init__(
        self,
        idx: int,
        sub_idx: int,
        size: int,
        fields: Dict[str, Tuple[int, int] | Tuple[int, int, int]],
    ):
        self.idx, self.sub_idx, self.size, self.fields = idx, sub_idx, size, fields

    def field_val(self, obj_val: int, field_name: str) -> int:
        mask, shift, *bin_val = self.fields[field_name]
        field_val = (obj_val & mask) >> shift
        return field_val if len(bin_val) == 0 else field_val == bin_val[0]

    def field_update(self, obj_val: int, field_name: str, field_val: int) -> int:
        mask, shift, *bin_val = self.fields[field_name]
        if len(bin_val) != 0:
            field_val = bin_val[0]
        return (obj_val & ~mask) | (field_val << shift)


class FestoDevice:
    def __init__(self, port: str):
        self.serial = self._open_serial_connection(port)
        self.objects: Dict[str, FestoCanObject] = self._build_objects()
        self.last_read = (0x0, 0x0, 0x0)

    def _build_objects(self):
        return {
            name: FestoCanObject(idx, sub_idx, size, fields)
            for name, idx, sub_idx, size, fields in CAN_OBJECTS_DESCRIPTIONS
        }

    def _set_units(self):
        motor_to_gear = Fraction(MOTOR_TO_AXIS_GEAR_RATION)
        ticks_per_rev = Fraction(65536)
        axis_feed = Fraction(AXIS_FEED_MM_PER_REV)
        fraction = (motor_to_gear * ticks_per_rev) / axis_feed
        num, denom = int(fraction.numerator), (fraction.denominator)
        self.sdo_write("position_factor_numerator", num)
        self.sdo_write("position_factor_divisor", denom)

    def _open_serial_connection(self, port: str):
        self.ser = serial.Serial(
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

    def _write(self, msg: str):
        self.ser.write((msg + "\r").encode())

    def _read(self) -> str:
        return self.ser.read_until(b"\r")[:-1].decode()

    def do_homing(self, force=False) -> Callable:
        print("start homing")
        if not force and self.sdo_read("manufacturer_status_word1", "is_referenced"):
            return lambda: None
        self._write("OW:1:0010:00000022")
        res = self._read()
        assert res == "OK!"

        def is_done():
            print("Wait for homing to complete.")
            while True:
                time.sleep(0.1)
                if self.sdo_read("manufacturer_status_word1", "is_referenced"):
                    print("Homing Completed.")
                    break

        return is_done

    def do_program(self, prog_id: int) -> Callable:
        self._write(f"OW:1:0010:{prog_id:04x}0021")
        rs = self._read()
        assert rs == "OK!", rs

        def wait_complete():
            while not self.sdo_read("status", "target_reached"):
                time.sleep(0.1)

        return wait_complete

    def _sdo_access(self, obj: FestoCanObject, wval: int | None) -> int:
        if wval is None:  # read
            rq = f"?{obj.idx:04x}{obj.sub_idx:02x}"
        else:
            rq = f"={obj.idx:04x}{obj.sub_idx:02x}:{wval:0{obj.size * 2}x}"
        self._write(rq)
        rs = self._read()
        if rs[0] == "!" and rs[-1] != "!":
            raise RuntimeError(SdoAccessError(int(rs[1:9], base=16)))
        elif rs[0] != "=":
            raise RuntimeError(f"Failed sdo access: rq:'{rq}', rs:'{rs}'")
        rs_idx, rs_sub_idx, rs_val = (
            int(rs[1:5], base=16),
            int(rs[5:7], base=16),
            int(rs[8:], base=16),
        )
        if rs_idx != obj.idx or rs_sub_idx != obj.sub_idx:
            raise RuntimeError(f"Invalid sdo access response: rq:'{rq}', rs:'{rs}'")
        if self.last_read != (obj.idx, obj.sub_idx, rs_val):
            # print(f"update: {obj.idx:04x},{obj.sub_idx:02x},{rs_val:0b}")
            self.last_read = (obj.idx, obj.sub_idx, rs_val)
        return rs_val

    @overload
    def sdo_read(self, obj_name: str) -> int: ...

    @overload
    def sdo_read(self, obj_name: str, first_field: str) -> int: ...

    @overload
    def sdo_read(
        self, obj_name: str, first_field: str, second_field: str, *args: str
    ) -> List[int]: ...

    def sdo_read(
        self,
        obj_name: str,
        first_field: None | str = None,
        second_field: None | str = None,
        *args: str,
    ) -> int | List[int]:
        obj = self.objects[obj_name]
        val = self._sdo_access(obj, None)
        if first_field and not second_field:
            return obj.field_val(val, first_field)
        elif first_field and second_field:
            return [
                obj.field_val(val, first_field),
                obj.field_val(val, second_field),
            ] + [obj.field_val(val, fname) for fname in args]
        return val

    def sdo_write(self, obj_name: str, val: int) -> int:
        obj = self.objects[obj_name]
        return self._sdo_access(obj, val)

    def sdo_update(self, obj_name: str, **kwargs: int) -> int:
        obj = self.objects[obj_name]
        val = self._sdo_access(obj, None)
        for fname, fval in kwargs.items():
            val = obj.field_update(val, fname, fval)
        return self.sdo_write(obj_name, val)

    def read(self, obj_name: str) -> int:
        def decode_response(res: str) -> Tuple[int, int, int]:
            eq, res_obj_idx, res_obj_sub_idx, val = (
                res[0],
                res[1:5],
                res[5:7],
                res[7:-1],
            )
            assert eq == "="
            assert len(res) == 4 or len(res) == 8
            return (
                int(res_obj_idx, base=16),
                int(res_obj_sub_idx, base=16),
                int(val, base=16),
            )

        assert obj_name in self.objects.keys()
        req_obj_idx, req_obj_sub_idx = self.objects[obj_name].idx, 0
        self.ser.write(f"?{req_obj_idx:04x}{req_obj_sub_idx:02x}\r".encode())
        res_obj_idx, res_obj_sub_idx, val = decode_response(
            self.ser.read_until(b"\r").decode()
        )
        assert res_obj_idx == req_obj_idx and res_obj_sub_idx == req_obj_sub_idx
        return val

    def move_to(self, pos: float, v: float = 1, a: float = 0.1, relative=False):
        self._set_units()
        to_units = lambda x: int(x)  # int(x * 10_000)
        pos, v, a = to_units(pos), to_units(v), to_units(a)
        target_mode = 1  # profile position mode
        current_mode = self.sdo_read("modes_of_operation_display")
        start_time = time.time()

        def timed_out():
            return time.time() - start_time > 1

        self.sdo_update("modes_of_operation", profile_position_mode=True)
        while current_mode != target_mode and not timed_out():
            time.sleep(0.1)

        current_mode = self.sdo_read("modes_of_operation_display")
        assert current_mode == target_mode, f"wrong operation mode: {current_mode}"

        assert v == ctypes.c_uint32(v).value
        assert a == ctypes.c_uint32(a).value
        assert pos == ctypes.c_int32(pos).value

        self.sdo_write("target_position", pos)
        self.sdo_write("profile_velocity", v)
        self.sdo_write("profile_acceleration", a)
        self.sdo_write("profile_deceleration", a)

        self.sdo_update("control", relative=1 if relative else 0)
        self.sdo_update("control", new_set_point=0)
        self.sdo_update("control", new_set_point=1)

        while not self.sdo_read("status", "set_point_acknowledge"):
            time.sleep(0.1)

        print("Acknowledged")

        while not self.sdo_read("status", "target_reached"):
            time.sleep(0.1)

        print("Reached")

    def can_start(self):
        state_machine = Dsp402StateMachine(self)
        state_machine.initialize_drive()


@enum.unique
class Dsp402State(enum.Enum):
    NOT_READY_TO_SWITCH_ON = "Not Ready to Switch On"
    SWITCH_ON_DISABLED = "Switch On Disabled"
    READY_TO_SWITCH_ON = "Ready to Switch On"
    SWITCHED_ON = "Switched On"
    OPERATION_ENABLED = "Operation Enabled"
    FAULT = "Fault"
    FAULT_REACTION_ACTIVE = "Fault Reaction Active"
    QUICK_STOP_ACTIVE = "Quick Stop Active"


class Dsp402StateMachine:
    def __init__(self, device: FestoDevice, timeout_s: float = 1.0):
        self.device = device
        self.timeout_s = timeout_s
        self.current_state = self._determine_current_state()

    def _determine_current_state(self) -> Dsp402State:
        """Reads the status word and maps it to a Dsp402State."""
        status_val = self.device.sdo_read("status")

        if self.device.objects["status"].field_val(status_val, "fault"):
            return Dsp402State.FAULT
        if self.device.objects["status"].field_val(status_val, "fault_reaction_active"):
            return Dsp402State.FAULT_REACTION_ACTIVE
        if self.device.objects["status"].field_val(status_val, "operation_enable"):
            return Dsp402State.OPERATION_ENABLED
        if self.device.objects["status"].field_val(status_val, "switched_on"):
            return Dsp402State.SWITCHED_ON
        if self.device.objects["status"].field_val(status_val, "ready_to_switch_on"):
            return Dsp402State.READY_TO_SWITCH_ON
        if self.device.objects["status"].field_val(status_val, "switch_on_disabled"):
            return Dsp402State.SWITCH_ON_DISABLED
        if self.device.objects["status"].field_val(
            status_val, "not_ready_to_switch_on"
        ):
            return Dsp402State.NOT_READY_TO_SWITCH_ON

        raise RuntimeError(f"Unknown device state from status word: {status_val:#06x}")

    def _wait_for_state(self, target_state: Dsp402State) -> bool:
        """Waits for the device to reach the target_state."""
        start_time = time.time()
        while time.time() - start_time < self.timeout_s:
            self.current_state = self._determine_current_state()
            if self.current_state == target_state:
                return True
            if self.current_state == Dsp402State.FAULT:
                raise RuntimeError(
                    f"Device entered FAULT state while waiting for {target_state.name}"
                )
            time.sleep(0.1)  # Polling interval
        raise TimeoutError(
            f"Timeout waiting for state {target_state.name}. Last state: {self.current_state.name}"
        )

    def _send_control_command(self, command_field: str):
        """Sends a control command using sdo_update."""
        # The `control_obj` fields are defined such that passing `1` as the value
        # will select the predefined binary pattern for that command.
        self.device.sdo_update("control", **{command_field: 1})

    def initialize_drive(self):
        """Guides the drive through the DSP402 initialization sequence."""
        print(f"Initial state: {self.current_state.name}")

        # Check if limit switches are configured correctly - avoid hitting endpoints
        if not self.device.sdo_read("limit_switch_polarity", "normally_closed"):
            raise RuntimeError("Expect normally closed limit switches.")

        # Enable CAN control
        self.device.sdo_write(
            "enable_logic", 2
        )  # physical signal + CAN logic for drive to function
        if not self.device.sdo_read("status", "remote"):
            time.sleep(0.2)  # Give it a moment
            if not self.device.sdo_read("status", "remote"):
                raise RuntimeError(
                    "Device did not switch to remote mode after enabling logic."
                )
        print("Remote mode enabled.")

        # Refresh state
        self.current_state = self._determine_current_state()
        print(f"State after enabling remote: {self.current_state.name}")

        # Clear Fault state first if required.
        if self.current_state == Dsp402State.FAULT:
            print("Device in FAULT state. Attempting fault reset...")
            # Transition reset_fault from 1 -> 0 clears fault, set 1, wait, set 0
            self.device.sdo_update("control", result_fault=1)
            time.sleep(0.1)
            self.device.sdo_update("control", result_fault=0)
            time.sleep(0.1)
            current_control_val = self.device.sdo_read("control")
            updated_control_val = self.device.objects["control"].field_update(
                current_control_val, "reset_fault", 0
            )
            self.device.sdo_write("control", updated_control_val)

            self._wait_for_state(Dsp402State.SWITCH_ON_DISABLED)
            print(f"Fault reset successful. State: {self.current_state.name}")

        # Transition: Switch On Disabled -> Ready to Switch On
        if self.current_state == Dsp402State.SWITCH_ON_DISABLED:
            print("State: SWITCH_ON_DISABLED. Sending SHUTDOWN command...")
            self._send_control_command(
                "shutdown"
            )  # Transition 2 (Controlword: 0xxx x110)
            self._wait_for_state(Dsp402State.READY_TO_SWITCH_ON)
            print(f"Transitioned to: {self.current_state.name}")

        # Transition: Ready to Switch On -> Switched On
        if self.current_state == Dsp402State.READY_TO_SWITCH_ON:
            print("State: READY_TO_SWITCH_ON. Sending SWITCH ON command...")
            self._send_control_command(
                "switch_on"
            )  # Transition 3 (Controlword: 0xxx 0111)
            self._wait_for_state(Dsp402State.SWITCHED_ON)
            print(f"Transitioned to: {self.current_state.name}")

        # Transition: Switched On -> Operation Enabled
        if self.current_state == Dsp402State.SWITCHED_ON:
            print("State: SWITCHED_ON. Sending ENABLE OPERATION command...")
            self._send_control_command(
                "enable_operation"
            )  # Transition 4 (Controlword: 0xxx 1111)
            self._wait_for_state(Dsp402State.OPERATION_ENABLED)
            print(f"Transitioned to: {self.current_state.name}")

        if self.current_state != Dsp402State.OPERATION_ENABLED:
            raise RuntimeError(
                f"Failed to initialize drive. Final state: {self.current_state.name}"
            )

        print("Drive initialized successfully to OPERATION_ENABLED.")


if __name__ == "__main__":
    ctrl = FestoDevice("COM3")
    ctrl.can_start()
    wait_for_homing = ctrl.do_homing()
    wait_for_homing()
    ctrl.move_to(500, a=100, v=100, relative=False)
