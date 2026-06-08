from isaaclab.app import AppLauncher
from isaacsim import SimulationApp
import argparse


# i generally separate the app launching feature for ease of use and readability
def launch_app() -> tuple:
    """
    Launch the IsaacSim app
    """
    # pass arguments to the launcher
    # even if i do not add any arguments, default ones still exist to run isaaclab
    # i.e. you can specify the number of environments, or pass the --enable_cameras flag to allow use of camera sensors
    parser = argparse.ArgumentParser(description="6-dof biped walking trainer")
    AppLauncher.add_app_launcher_args(parser)
    
    args_cli = parser.parse_args()
    
    # launch app with given args
    app_launcher = AppLauncher()
    sim_app = app_launcher.app
    # we may use the arguments, so it's important to pass them
    return sim_app, args_cli