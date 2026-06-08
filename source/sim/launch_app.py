from isaaclab.app import AppLauncher
from isaacsim import SimulationApp
import argparse


def launch_app() -> tuple:
    """
    Launch the IsaacSim app
    """
    # pass arguments to the launcher
    parser = argparse.ArgumentParser(description="6-dof biped walking trainer")
    AppLauncher.add_app_launcher_args(parser)
    
    args_cli = parser.parse_args()
    
    # launch app with given args
    app_launcher = AppLauncher()
    sim_app = app_launcher.app
    
    return sim_app, args_cli