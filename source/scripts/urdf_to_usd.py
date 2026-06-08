import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


# urdf (universal robot description format) must be converted to pixar's usd (universal scene descriptor), which is what isaac uses
# isaaclab requires confiurations of most objects
cfg = UrdfConverterCfg(
    asset_path="/source/biped/assembly_1/urdf/assembly_1.urdf", # this is the vanilla urdf path of the robot
    usd_dir="/source", # this is the destination folder
    usd_file_name="biped.usd",
    fix_base=False, # whether to "bolt" the base of the robot to the ground; we do not want this, but something like a robotic arm might
    make_instanceable=True,
)
converter = UrdfConverter(cfg)