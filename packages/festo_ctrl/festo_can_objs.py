import enum

CAN_OBJECT_DESCRIPTIONS = []

@enum.unique
class SdoAccessError(enum.Enum):
    ProtocolError_ToggleBit = 0x05030000
    ProtocolError_ClientServerCommandSpecifierInvalid = 0x05040001
    AccessFaulty_HardwareProblem = 0x06060000  # *1)
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
    DataTransmissionOrStorageFailure = 0x08000020  # *1)
    DataTransmissionRegulatorLocal = 0x08000021
    DataTransmissionRegulatorIncorrectState = 0x08000022  # *3)
    NoObjectDictionaryAvailable = 0x08000023  # *2


# name, index, subindex, wordsize, fields[mask, shift, binary_value]
control_obj = ("control", 0x6040, 0, 2, {
    "shutdown"              : (0x7, 0, 0b110),
    "switch_on"             : (0xF, 0, 0b111),
    "disable_voltage"       : (0x2, 1, 0b0),
    "quick_stop"            : (0x6, 1),
    "disable_operation"     : (0xF, 0, 0b0111),
    "enable_operation"      : (0xF, 0, 0b1111),
    "new_set_point"         : (0x10, 4),  # In profile position mode
    "start_homing_operation": (0x10, 4),  # In homing mode
    "enable_ip_mode"        : (0x10, 4),  # In interpolated position mode
    "change_set_immediately": (0x20, 5),  # Only in profile position mode
    "relative"     : (0x40, 6),  # Only inprofile position mode
    "reset_fault"           : (0x80, 7),
    "halt"                  : (0x100, 8),
}
               )  # RW

status_obj = ("status", 0x6041, 0, 2, {
    "not_ready_to_switch_on": (0x4f, 0, 0x0),
    "switch_on_disabled"    : (0x4f, 0, 0x40),
    "ready_to_switch_on"    : (0x6f, 0, 0x21),
    "switched_on"           : (0x6f, 0, 0x23),
    "operation_enable"      : (0x6f, 0, 0x27),
    "quick_stop_active"     : (0x6f, 0, 0x7),
    "fault_reaction_active" : (0x4f, 0, 0xF),
    "fault"                 : (0x4f, 0, 0x8),
    "voltage_enabled"       : (0x10, 4),
    "warning"               : (0x80, 7),
    "drive_is_moving"       : (0x100, 8),
    "remote"                : (0x200, 9),
    "target_reached"        : (0x400, 10),  # In profile position & velocity mode
    "internal_limit_active" : (0x800, 11),
    "set_point_acknowledge" : (0x1000, 12),  # In profile position mode
    "speed_0"               : (0x1000, 12),  # In profile velocity mode
    "homing_attained"       : (0x1000, 12),  # In homing mode
    "ip_mode_active"        : (0x1000, 12),  # In interpolated position mode
    "following_error"       : (0x2000, 13),  # In profile velocity mode
    "homing_error"          : (0x2000, 13),  # in homing mode
    "manufacturer_statusbit": (0x4000, 14),
    "drive_refernced"       : (0x8000, 15),
})  # RO

enable_logic_obj = ("enable_logic", 0x6510, 0x10, 2, {
    "enable_logic": (0x3, 0)  #
})

modes_of_operation_obj = ("modes_of_operation", 0x6060, 0, 1, {
    "profile_position_mode"     : (0xFF, 0, 1),
    "profile_velocity_mode"     : (0xFF, 0, 3),
    "torque_profile_mode"       : (0xFF, 0, 4),
    "homing_mode"               : (0xFF, 0, 6),
    "interpolated_position_mode": (0xFF, 0, 7),
})

modes_of_operation_display_obj = ("modes_of_operation_display", 0x6061, 0, 1, {
})

limit_switch_polarity_obj = ("limit_switch_polarity", 0x6510, 0x11, 2, {
    "normally_closed" : (0b1, 0, 0),
    "normally_open"   : (0b1, 0, 1)
})

target_position_obj = ("target_position", 0x607A, 0, 4, {
    "value" : (0xFFFFFFFF, 0)
})

profile_velocity_obj = ("profile_velocity", 0x6081, 0, 4, {
    "value" : (0xFFFFFFFF, 0)
})

profile_acceleration_obj = ("profile_acceleration", 0x6083, 0, 4, {
    "value" : (0xFFFFFFFF, 0)
})

profile_deceleration_obj = ("profile_deceleration", 0x6084, 0, 4, {
    "value" : (0xFFFFFFFF, 0)
})

position_factor_numerator_obj = ("position_factor_numerator", 0x6093, 1, 4, {
    "value": (0xFFFFFFFF, 0)
})

position_factor_divisor_obj = ("position_factor_divisor", 0x6093, 2, 4, {
    "value": (0xFFFFFFFF, 0)
})

manufacturer_statusword_1 = ("manufacturer_status_word1", 0x2000, 1, 4,
    {
        "is_referenced" : (0b1, 0),
        "communication_valid" : (0b1, 1),
        "ready_for_enable" : (0b1, 2),
    })

CAN_OBJECTS_DESCRIPTIONS = [control_obj, status_obj, enable_logic_obj, modes_of_operation_obj, modes_of_operation_display_obj,
    limit_switch_polarity_obj, target_position_obj, profile_velocity_obj, profile_acceleration_obj,
    profile_deceleration_obj, manufacturer_statusword_1, position_factor_numerator_obj, position_factor_divisor_obj]
