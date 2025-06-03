import enum
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from festo_ctrl.stage_controller import FestoDevice


class SdoAccessError(enum.Enum):
    ProtocolError_ToggleBit = 0x05030000
    ProtocolError_ClientServerCommandSpecifierInvalid = 0x05040001
    AccessFaulty_HardwareProblem = 0x06060000
    AccessTypeNotSupported = 0x06010000
    ReadAccessOnly = 0x06010001
    WriteAccessOnly = 0x06010002
    ObjectDoesNotExist = 0x06020000
    ObjectMustNotBeEnteredInPDO = 0x06040041
    PDOLengthExceeded = 0x06040042
    GeneralParameterError = 0x06040043
    InternalVariableOverflow = 0x06040047
    ProtocolError_ServiceParameterLengthMismatch = 0x06070010
    ProtocolError_ServiceParameterTooLarge = 0x06070012
    ProtocolError_ServiceParameterTooSmall = 0x06070013
    SubindexDoesNotExist = 0x06090011
    DataExceedsObjectRange = 0x06090030
    DataTooLarge = 0x06090031
    DataTooSmall = 0x06090032
    UpperLimitLessThanLower = 0x06090036
    DataTransmissionOrStorageFailure = 0x08000020
    DataTransmissionRegulatorLocal = 0x08000021
    DataTransmissionRegulatorIncorrectState = 0x08000022
    NoObjectDictionaryAvailable = 0x08000023


@dataclass(frozen=True)
class ValueField:
    mask: int
    shift: int

    def read(self, raw_object_value: int) -> int:
        return (raw_object_value & self.mask) >> self.shift

    def write(self, current_object_value: int, value_to_set: int) -> int:
        return (current_object_value & ~self.mask) | (
            (value_to_set << self.shift) & self.mask
        )


@dataclass(frozen=True)
class FlagField:
    mask: int
    shift: int
    pattern: int  # The specific bit pattern that means "true" for this flag

    def read(self, raw_object_value: int) -> bool:
        return ((raw_object_value & self.mask) >> self.shift) == self.pattern

    def write(self, current_object_value: int, value_to_set: int) -> int:
        # clear bits, in case of flag disable, bits are 0, not inverted pattern, ...
        new_value = current_object_value & ~self.mask
        if value_to_set:
            new_value |= (self.pattern << self.shift) & self.mask
        return new_value


FieldDefinition = Union[ValueField, FlagField]


class CanObjectBase:
    INDEX: int
    SUB_INDEX: int
    SIZE: int
    _object_name: str

    def __init__(self, device: "FestoDevice"):
        self._device = device

    def _get_field_def(self, field_name: str) -> FieldDefinition:
        field_def = getattr(self.__class__, field_name, None)
        if not isinstance(field_def, (ValueField, FlagField)):
            raise AttributeError(
                f"'{field_name}' is not a valid field for {self.__class__.__name__}"
            )
        return field_def

    def read(self, field: str | None = None) -> int | bool:
        raw_val = self._device.object_read(self)
        if field is None:
            return raw_val
        field_def = self._get_field_def(field)
        return field_def.read(raw_val)

    def write(self, value: int) -> int:
        return self._device.object_write(self, value)

    def update(self, **fields: int) -> int:
        new_value = self._device.object_read(self)

        for field_name, value_to_set in fields.items():
            field_def = self._get_field_def(field_name)
            new_value = field_def.write(new_value, value_to_set)

        return self._device.object_write(self, new_value)

    @classmethod
    def field_names(cls) -> list[str]:
        return [
            name
            for name, value in inspect.getmembers(cls)
            if isinstance(value, (ValueField, FlagField))
        ]


class Control(CanObjectBase):
    INDEX, SUB_INDEX, SIZE, _object_name = 0x6040, 0, 2, "control"
    # fmt: off
    shutdown              : FlagField = FlagField(mask=0x7,   shift=0, pattern=0b110)
    switch_on             : FlagField = FlagField(mask=0xF,   shift=0, pattern=0b111)
    disable_voltage       : FlagField = FlagField(mask=0x2,   shift=1, pattern=0b0)
    quick_stop            : FlagField = FlagField(mask=0xF,   shift=1, pattern=0b01)
    disable_operation     : FlagField = FlagField(mask=0xF,   shift=0, pattern=0b0111)
    enable_operation      : FlagField = FlagField(mask=0xF,   shift=0, pattern=0b1111)
    new_set_point         : FlagField = FlagField(mask=0x10,  shift=4, pattern=1)
    start_homing_operation: FlagField = FlagField(mask=0x10,  shift=4, pattern=1)
    enable_ip_mode        : FlagField = FlagField(mask=0x10,  shift=4, pattern=1)
    change_set_immediately: FlagField = FlagField(mask=0x20,  shift=5, pattern=1)
    relative              : FlagField = FlagField(mask=0x40,  shift=6, pattern=1)
    reset_fault           : FlagField = FlagField(mask=0x80,  shift=7, pattern=1)
    halt                  : FlagField = FlagField(mask=0x100, shift=8, pattern=1)
    # fmt: on


class Status(CanObjectBase):
    INDEX, SUB_INDEX, SIZE, _object_name = 0x6041, 0, 2, "status"
    # fmt: off
    # DSP402 State Flags
    not_ready_to_switch_on: FlagField = FlagField(mask=0x4F, shift=0, pattern=0x00)
    switch_on_disabled    : FlagField = FlagField(mask=0x4F, shift=0, pattern=0x40)
    ready_to_switch_on    : FlagField = FlagField(mask=0x6F, shift=0, pattern=0x21)
    switched_on           : FlagField = FlagField(mask=0x6F, shift=0, pattern=0x23)
    operation_enable      : FlagField = FlagField(mask=0x6F, shift=0, pattern=0x27)
    quick_stop_active     : FlagField = FlagField(mask=0x6F, shift=0, pattern=0x07)
    fault_reaction_active : FlagField = FlagField(mask=0x4F, shift=0, pattern=0x0F)
    fault                 : FlagField = FlagField(mask=0x4F, shift=0, pattern=0x08)

    # More Status bits
    voltage_enabled       : FlagField = FlagField(mask=(1<<4),  shift=4,  pattern=1)
    warning               : FlagField = FlagField(mask=(1<<7),  shift=7,  pattern=1)
    drive_is_moving       : FlagField = FlagField(mask=(1<<8),  shift=8,  pattern=1)
    remote                : FlagField = FlagField(mask=(1<<9),  shift=9,  pattern=1)
    target_reached        : FlagField = FlagField(mask=(1<<10), shift=10, pattern=1)
    internal_limit_active : FlagField = FlagField(mask=(1<<11), shift=11, pattern=1)

    # Bit 12 - mode dependent flags
    set_point_acknowledge : FlagField = FlagField(mask=(1<<12), shift=12, pattern=1)
    speed_0               : FlagField = FlagField(mask=(1<<12), shift=12, pattern=1)
    homing_attained       : FlagField = FlagField(mask=(1<<12), shift=12, pattern=1)
    ip_mode_active        : FlagField = FlagField(mask=(1<<12), shift=12, pattern=1)

    # Bit 13 - mode dependent flags
    following_error       : FlagField = FlagField(mask=(1<<13), shift=13, pattern=1)
    homing_error          : FlagField = FlagField(mask=(1<<13), shift=13, pattern=1)

    manufacturer_statusbit: FlagField = FlagField(mask=(1<<14), shift=14, pattern=1)
    drive_referenced      : FlagField = FlagField(mask=(1<<15), shift=15, pattern=1)
    # fmt: on


class EnableLogic(CanObjectBase):
    INDEX, SUB_INDEX, SIZE, _object_name = 0x6510, 0x10, 2, "enable_logic"
    enable_logic: ValueField = ValueField(mask=0x3, shift=0)


class ModesOfOperation(CanObjectBase):
    INDEX, SUB_INDEX, SIZE, _object_name = 0x6060, 0, 1, "modes_of_operation"
    # fmt: off
    profile_position_mode     : FlagField = FlagField(mask=0xFF, shift=0, pattern=1)
    profile_velocity_mode     : FlagField = FlagField(mask=0xFF, shift=0, pattern=3)
    torque_profile_mode       : FlagField = FlagField(mask=0xFF, shift=0, pattern=4)
    homing_mode               : FlagField = FlagField(mask=0xFF, shift=0, pattern=6)
    interpolated_position_mode: FlagField = FlagField(mask=0xFF, shift=0, pattern=7)
    # fmt: on


class ModesOfOperationDisplay(CanObjectBase):
    INDEX, SUB_INDEX, SIZE, _object_name = 0x6061, 0, 1, "modes_of_operation_display"


class LimitSwitchPolarity(CanObjectBase):
    INDEX, SUB_INDEX, SIZE, _object_name = 0x6510, 0x11, 2, "limit_switch_polarity"
    # fmt: off
    normally_closed: FlagField = FlagField(mask=0b1, shift=0, pattern=0)
    normally_open:   FlagField = FlagField(mask=0b1, shift=0, pattern=1)
    # fmt: on


class TargetPosition(CanObjectBase):
    INDEX, SUB_INDEX, SIZE, _object_name = 0x607A, 0, 4, "target_position"
    value: ValueField = ValueField(mask=0xFFFFFFFF, shift=0)


class ProfileVelocity(CanObjectBase):
    INDEX, SUB_INDEX, SIZE, _object_name = 0x6081, 0, 4, "profile_velocity"
    value: ValueField = ValueField(mask=0xFFFFFFFF, shift=0)


class ProfileAcceleration(CanObjectBase):
    INDEX, SUB_INDEX, SIZE, _object_name = 0x6083, 0, 4, "profile_acceleration"
    value: ValueField = ValueField(mask=0xFFFFFFFF, shift=0)


class ProfileDeceleration(CanObjectBase):
    INDEX, SUB_INDEX, SIZE, _object_name = 0x6084, 0, 4, "profile_deceleration"
    value: ValueField = ValueField(mask=0xFFFFFFFF, shift=0)


class ManufacturerStatusWord1(CanObjectBase):
    INDEX, SUB_INDEX, SIZE, _object_name = 0x2000, 1, 4, "manufacturer_status_word1"
    # fmt: off
    is_referenced:       FlagField = FlagField(mask=(1 << 0), shift=0, pattern=1)
    communication_valid: FlagField = FlagField(mask=(1 << 1), shift=1, pattern=1)
    ready_for_enable:    FlagField = FlagField(mask=(1 << 2), shift=2, pattern=1)
    # fmt: on


class PositionFactorNumerator(CanObjectBase):
    INDEX, SUB_INDEX, SIZE, _object_name = 0x6093, 1, 4, "position_factor_numerator"
    value: ValueField = ValueField(mask=0xFFFFFFFF, shift=0)


class PositionFactorDivisor(CanObjectBase):
    INDEX, SUB_INDEX, SIZE, _object_name = 0x6093, 2, 4, "position_factor_divisor"
    value: ValueField = ValueField(mask=0xFFFFFFFF, shift=0)
