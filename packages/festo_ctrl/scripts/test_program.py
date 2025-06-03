from festo_ctrl.stage_controller import FestoController

if __name__ == "__main__":
    ctrl = FestoController("COM3")
    ctrl.can_start()
    ctrl.do_program(10)
    ctrl.wait_for_stopped()
