# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # This is a kinematic example for an articulated robot arm with 6 joints
# %% [markdown]
# As a basic example think of a cube, that is to be processed in several steps to transfer it from its initial state to a defined target state.
# 
# Initial state of cube:
# 
# <center>
# <img src="images/initial_cube.PNG" width="400" alt="initial state" >
# </center>
# 
# After the 1st processing step the cube is supposed to look like this:
# 
# 1st intermediate state of cube:
# 
# <center>
# <img src="images/cut_cube.PNG" width="400" alt="cut state">
# </center>
# 
# To calculate the tool path for the robot to execute this processing, a CAD/CAM system compares these two CAD-files and determines the tool path for a specific milling tool, that is required to execute this processing step. In this case, this tool path would just be a straight line in Cartesian space. The motion commands needed by the robot control to execute the planned tool path are stored in a nc-file. This is send to the robot control and there transformed from operating space (Cartesian) to joint space (angular). Let's do this step by step, until we know the joint angles.
# %% [markdown]
# First of all the kinematics of the robot are initialized specifying its link lengths.

# %%
import RobotClass as rc
import warnings
warnings.filterwarnings('ignore')

# these are part of the Denavit-Hartenber-parameter that define the kinematics of a manipulator

# nominal link lengths
dh_parameter = {
    'l11': 550,
    'l12': 450,
    'l2': 860,
    'l3': 210,
    'l4': 762,
    'l5': 0,
    'l6': 210
}

kin = rc.kinematic(dh_parameter)

# %% [markdown]
# Then the $n$ positions ($x$, $y$, $z$) and orientations ($\alpha$, $\beta$, $\gamma$) of the Tool Center Point (TCP) of the robot for the needed linear motion in Cartesian space are obtained here by linear interpolation between the start and end points (these coordinates would otherwise be calculated by the CAD/CAM system).

# %%
tcp_xyz_abg = kin.linear_interpolation(
    [1000, 0, 2000], [1000, 100, 2000], 
    [90, 0, 90], [90, 0, 90], 
    10
    ) 

# %% [markdown]
# Now let's have a look at the tool path.

# %%
kin.plot_cartesian_space([tcp_xyz_abg])

# %% [markdown]
# Not surprisingly, it's a perfect straight line in Cartesian space. Now applying the inverse kinematic transformation the joint angles corresponding to the TCP path are determined.

# %%
joint_angles = kin.inverse(tcp_xyz_abg)

# %% [markdown]
# Again, let's have a look at them.

# %%
kin.plot_joint_space(joint_angles)

# %% [markdown]
# Now it is important to keep in mind, that these joint angles correspond to the nominal robot link lengths, that we specified earlier. If we redefine them slightly and then direct transform these new (actual) joint angles to Cartesian space (as would be done by the physical real robot), the actual tool path would appear different.

# %%
# actual link lengths
dh_parameter = {
    'l11': 550+3,
    'l12': 450-5,
    'l2': 860+5,
    'l3': 210+4,
    'l4': 762+7,
    'l5': 0+2,
    'l6': 210+3
}

kin = rc.kinematic(dh_parameter)


# %%
tcp_xyz_abg1 = kin.direct(joint_angles)
kin.plot_cartesian_space([tcp_xyz_abg, tcp_xyz_abg1])


