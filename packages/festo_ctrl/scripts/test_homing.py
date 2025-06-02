from festo_ctrl.stage_controller import FestoController

if __name__ == "__main__":
    ctrl = FestoController("COM3")
    ctrl.can_start()
    wait_for_homing = ctrl.do_homing()
    wait_for_homing()
