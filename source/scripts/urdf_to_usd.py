import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


cfg = UrdfConverterCfg(
    asset_path="/source/biped/assembly_1/urdf/assembly_1.urdf",
    usd_dir="/source",
    usd_file_name="biped.usd",
    fix_base=False,
    make_instanceable=True,
)
converter = UrdfConverter(cfg)