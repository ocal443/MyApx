from festo_ctrl.stage_controller import FestoController

if __name__ == "__main__":
    ctrl = FestoController("COM3")
    ctrl.can_start()
    ctrl.move_to(0, a=300, v=1000, relative=False)
