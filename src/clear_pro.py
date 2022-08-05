import roboticstoolbox as rtb
from spatialmath import *
from spatialmath.base import *
import numpy as np
from ctypes import cdll, POINTER, c_double, byref
import os
import open3d as o3d
from stl import mesh
import copy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.widgets import Slider
from roboticstoolbox.backends.PyPlot import PyPlot
import lmfit as lf
import configparser
import time



# class to read, write and pertubate robot parameters
class Parameters(object):
    """reads, perturbates and writes robot parameters"""
    def __init__(self) -> None:
        pass

    def read_parameters(self, file, is_print=False):
        """reads robot model parameter from ini file and counts variable 
        parameters

        Parameters
        ----------
        file : string
            ini file path

        Returns
        -------
        lmfit parameter object
            robot model parameters
        """
        # read ini file
        config = configparser.ConfigParser()
        config.read(file)
        parameter_names = [
            'off1', 'off2', 'off3', 'off4', 'off5', 'off6',
            'd1', 'd2', 'd3', 'd4', 'd5', 'd6',
            'a1', 'a2', 'a3', 'a4', 'a5', 'a6',
            'alpha1', 'alpha2', 'alpha3','alpha4','alpha5', 'alpha6',
            'tooltx', 'toolty', 'tooltz', 'toolrx', 'toolry', 'toolrz',
            'basetx', 'basety', 'basetz', 'baserx', 'basery', 'baserz'
            ]
        # create lmfit parameter object
        parameters = lf.Parameters()
        for name in parameter_names:
            param = config.get('parameters' , name).split()
            parameters.add(name,
                float(param[0]), param[1]=='True', float(param[2]), float(param[3])
                )
        if is_print:
            # count variable parameters
            num_param_vary_true = 0
            for _, parameter in parameters.items():
                if parameter.vary == True:
                    num_param_vary_true += 1
            print('\n')
            print('Number of variable parameters ('+file+'):')
            print('------------------------------')
            print(num_param_vary_true)

        return parameters

    def contrast_parameters(self, ideal=None, real=None):
        """compares ideal and real parameters

        Parameters
        ----------
        ideal : lmfit parameter object, optional
            ideal parameters, by default None
        real : lmfit parameter object, optional
            real parameters, by default None
        """
        print('\n')
        print('Parameters ideal / real:')
        print('-----------------------------')
        for (key1, parameter1), (key2, parameter2) in zip(
            ideal.items(), real.items()):
            if parameter1.vary == True:
                print('{}: {:.3f} / {}: {:.3f}'.format(
                    key1, parameter1.value, key2, parameter2.value
                    )
                    )



# class to define ideal and real robots
class Robot(object):
    """holds all robot related information"""    
    def __init__(self, params, lims=None):
        """It is initiated with a set of d-h-parameters, joint angle limits and 
        a name.

        Parameters
        ----------
        params : lmfit parameter object
            d-h-parameter of robot (including base and tool transformation)
        lims : numpy array (6, 3)
            joint angle limits in rad
        name : string
            robot name
        """
        if lims is None:
            lims = np.array([
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
            ])
        # parameter
        p = params.valuesdict()
        # robot
        self.rtb_robot = rtb.robot.DHRobot(
            [
                rtb.robot.DHLink(
                    offset=p['off1'], d=p['d1'], a=p['a1'], alpha=p['alpha1'], qlim=[lims[0, 0], lims[0, 1]]
                    ),
                rtb.robot.DHLink(
                    offset=p['off2'], d=p['d2'], a= p['a2'], alpha=p['alpha2'], qlim=[lims[1, 0], lims[1, 1]]
                    ),
                rtb.robot.DHLink(
                    offset=p['off3'], d=p['d3'], a=-p['a3'], alpha=p['alpha3'], qlim=[lims[2, 0], lims[2, 1]]
                    ),
                rtb.robot.DHLink(
                    offset=p['off4'], d=p['d4'], a=p['a4'], alpha=p['alpha4'], qlim=[lims[3, 0], lims[3, 1]]
                    ),
                rtb.robot.DHLink(
                    offset=p['off5'], d=p['d5'], a=p['a5'], alpha=p['alpha5'], qlim=[lims[4, 0], lims[4, 1]]
                    ),
                rtb.robot.DHLink(
                    offset=p['off6'], d=p['d6'], a=p['a6'], alpha=p['alpha6'], qlim=[lims[5, 0], lims[5, 1]]
                    )
            ],
            name='Comau',
            tool = SE3(p['tooltx'], p['toolty'], p['tooltz']) * \
                SE3.RPY([p['toolrx'], p['toolry'], p['toolrz']], unit='rad', order='zyx'),
            base = SE3(p['basetx'], p['basety'], p['basetz']) * \
                SE3.RPY([p['baserx'], p['basery'], p['baserz']], unit='rad', order='zyx')
            )
        self.rtb_robot.addconfiguration('home', [0, 0, 0, 0, 0, 0])



# class to define workpiec
class Workpiece(object):
    """holds all workpiece related information"""
    def __init__(
        self, stock_file_path='stl/cube.stl', result_file_path='stl/result.stl',
        pose=SE3([0, 0, 0])
        ):
        """It is initiated with a file path to the stock mesh, a file path to
        the result mesh the workpiece pose in the robot base frame, the number 
        of sampling points for the creation of the point cloud data, the stock 
        pcd color and the result pcd color.

        Parameters
        ----------
        stock_file_path : str, optional
            file path to stock mesh, by default 'stl/cube.stl'
        result_file_path : str, optional
            file path for result mesh storage, by default 'stl/result.stl'
        pose : SE3, optional
            workpiece pose in robot base frame, by default SE3([0, 0, 0])
        npoints : int, optional
            number of sampling points, by default 1000
        stock_color : list, optional
            stock pcd color, by default [0, 1, 0]
        result_color : list, optional
            result pcd color, by default [1, 0, 0]
        """
        self.stock_file_path = stock_file_path
        self.result_file_path = result_file_path
        self.pose = pose
        if os.path.exists(stock_file_path):
            self.stock_mesh = o3d.io.read_triangle_mesh(stock_file_path)
            self.stock_mesh.compute_vertex_normals()
        else:
            print('\n')
            print('No stock file found')
        if os.path.exists(result_file_path):
            self.result_mesh = o3d.io.read_triangle_mesh(result_file_path)
            self.result_mesh.compute_vertex_normals()
        # else:
            # print('\n')
            # print('No result file found')
            
        print('\n')
        print('Workpiece class object instantiated')



# class to define tool
class Tool(object):
    """holds all tool related informations"""
    def __init__(self, diameter, flute_length):
        self.diameter = diameter
        self.flute_length = flute_length

        print('\n')
        print('Tool class object instantiated')



# class to define milling path
class Toolpath(object):
    """holds all toolpath related information"""

    def read_nc(self, file):
        """reads nc program from file

        Parameters
        ----------
        file : string
            file path to nc program

        Returns
        -------
        spatial math SE3 object
            4x4 homogenous transformation matrices of toolpath poses in workpiece frame
        """
        xcoords_list = []
        ycoords_list = []
        zcoords_list = []
        xangle_list = []
        yangle_list = []
        zangle_list = []
        with open(file, encoding='utf-16-le') as myfile:
            for line in myfile:
                xcoords_list.append(float(line.partition('X')[2].split()[0]))
                ycoords_list.append(float(line.partition('Y')[2].split()[0]))
                zcoords_list.append(float(line.partition('Z')[2].split()[0]))
                xangle_list.append(float(line.partition('SPA')[2].split()[0]))
                yangle_list.append(float(line.partition('SPB')[2].split()[0]))
                zangle_list.append(float(line.partition('SPC')[2].split()[0]))
        xcoords_np = np.asarray(xcoords_list)
        ycoords_np = np.asarray(ycoords_list)
        zcoords_np = np.asarray(zcoords_list)
        xangle_np = np.asarray(xangle_list)
        yangle_np = np.asarray(yangle_list)
        zangle_np = np.asarray(zangle_list)
        positions = np.vstack((xcoords_np, ycoords_np, zcoords_np)).T
        orientations = np.vstack((xangle_np, yangle_np, zangle_np)).T
        # calculate toolpath
        toolpath_w = [
            SE3(pos*1e-3) * SE3.RPY(ori, order='zyx', unit='deg') 
            for pos, ori in zip(positions, orientations)
            ] 
        print('\n')
        print('NC program read in')  

        return toolpath_w

    def transform_to_base(self, toolpath_w, workpiece):
        """transforms toolpath from workpiece to robot base frame

        Parameters
        ----------
        toolpath_w : spatial math SE3 object
            4x4 homogenous transformation matrices of toolpath poses in workpiece frame
        workpiece : clear_pro Workpiece object
            holds workpiece pose information

        Returns
        -------
        spatial math SE3 object
            4x4 homogenous transformation matrices of toolpath poses in robot base frame
        """
        toolpath_b = copy.deepcopy(toolpath_w)            
        toolpath_b = [workpiece.pose @ traj for traj in toolpath_w]

        print('\n')
        print('Toolpath transformed from workpiece frame to robot base frame')

        return toolpath_b



# class to define removal simulation settings
class Simulation(object):
    """enables removal simulations and identification of removed volume"""
    def __init__(self, parameters, workpiece, tool, toolpath, import_precision=100):
        """It is initiated with the Toolpath object, the Workpiece object
        and the import precision for ModuleWorks

        Parameters
        ----------
        toolpath : Toolpath class object
            holds the tool path informations
        workpiece : Workpiece class object
            holds the workpiece informations
        import_precision : int, optional
            import precision for Module Works, by default 10
        """
        self.workpiece = workpiece
        self.tool = tool
        self.toolpath = toolpath
        self.import_precision = import_precision
        self.robot_ideal = Robot(parameters[0])
        self.robot_real = Robot(parameters[1])
        self.joint_angles = np.asarray(
            [self.robot_ideal.rtb_robot.ikine_LM(traj).q for traj in self.toolpath]
            )
        
        print('\n')
        print('Simulation class object instantiated')

    def do_cut(
        self, parameters, is_print=False,
        is_delete=False, is_pcd=False, is_voxel_grid=False, npoints=1000, nvoxels=10
        ):
        """Executes the material removal simulation

        Parameters
        ----------
        parameters : lmfit parameter object
            robot kinematic parameters with which the simulation is to be executed
        is_plot : bool, optional
            determines whether or not results are plotted and additional information is printed, by default False
        is_delete : bool, optional
            determines whether the result mesh file is deleted after the simulation for clean up

        Returns
        -------
        o3d mesh
            material removal result mesh
        """
        robot = Robot(parameters)
        trajectory_rb = [robot.rtb_robot.fkine(ja) for ja in self.joint_angles]
        trajectory_w = [self.workpiece.pose.inv() @ traj for traj in trajectory_rb]
        # define tool path position and orientation
        tool_path_pos = np.asarray([traj.t*1e3 for traj in trajectory_w])
        tool_path_ori = np.asarray([traj.A[0:3, 2] for traj in trajectory_w])        
      
        # allocation array of double*
        # tool_path position
        tool_path_pos_in = (POINTER(c_double) * tool_path_pos.shape[0])()
        for i in range(tool_path_pos.shape[0]):
            # allocate arrays of double
            tool_path_pos_in[i] = (c_double * 3)()
            for j in range(3):
                tool_path_pos_in[i][j] = tool_path_pos[i, j]
        # tool_path orientation
        tool_path_ori_in = (POINTER(c_double) * tool_path_ori.shape[0])()
        for i in range(tool_path_ori.shape[0]):
            # allocate arrays of double
            tool_path_ori_in[i] = (c_double * 3)()
            for j in range(3):
                tool_path_ori_in[i][j] = tool_path_ori[i, j]

        # set file paths before current working directory is changed
        strings = self.workpiece.result_file_path.split('.')
        i = 0
        while os.path.exists(str(strings[0]+'_{}.stl').format(i)):
            i += 1
        result_path = str('../'+strings[0]+'_{}.stl').format(i)
        stock_file_path = str('../'+self.workpiece.stock_file_path).replace('/data', '').encode()
        result_file_path = str(result_path).replace('/data', '').encode()
        
        # change directory to dll folder so that other dll are found
        os.chdir(os.getcwd()+'/data/dll')    
        # invoke removal_simulation_dll
        mydll = cdll.LoadLibrary('removal_simulation_dll.dll')
        removal_simulation = mydll.removal_simulation
        # removal simulation(
        # const char* initial_workpiece_filename,
        # const char* final_workpiece_filename,
        # const double ** toolpath_position,
        # const double ** toolpath_orientation,
        # const int tool_diameter,
        # const int tool_flute_length,
        # const int import_precision,
        # const int rowcount
        # )
        start_time_rs = time.time()
        removal_simulation(
            stock_file_path,
            result_file_path,
            byref(tool_path_pos_in),
            byref(tool_path_ori_in),
            self.tool.diameter,
            self.tool.flute_length,
            self.import_precision,
            tool_path_pos.shape[0],
            is_print
        )
        if is_print:
            print('\n')
            print('calculation time (removal simulation): {:.3f}s'.format(
                time.time()-start_time_rs
            ))

        # change back to main directory
        os.chdir('../../')
        # save result mesh to object
        self.workpiece.result_mesh = o3d.io.read_triangle_mesh(result_path[3:])
        self.workpiece.result_mesh.compute_vertex_normals()
        
        # create pcd
        if is_pcd:
            start_time_pcd = time.time()
            self.workpiece.result_pcd = self.workpiece.result_mesh.sample_points_poisson_disk(number_of_points=npoints, init_factor=5)
            if is_print:
                print('\n')
                print('calculation time (sampling): {:.3f}s'.format(
                    time.time()-start_time_pcd
                ))
        # create voxel grid
        if is_voxel_grid:
            start_time_grid = time.time()
            v_size=round(
                max(self.workpiece.result_mesh.get_max_bound()-self.workpiece.result_mesh.get_min_bound())/
                nvoxels,
                3)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
                self.workpiece.result_mesh, voxel_size=v_size
                )
            self.workpiece.result_voxelgrid = voxel_grid
            if is_print:
                print('\n')
                print('calculation time (voxel grid): {:.3f}s'.format(
                    time.time()-start_time_grid
                ))
        else:
            self.workpiece.result_voxelgrid = None

        if is_delete:
            os.remove(result_path[3:])

        if is_print:
            print('\n')
            print('Material removal simulation executed')

        return self.workpiece.result_pcd, self.workpiece.result_mesh, self.workpiece.result_voxelgrid
       


# class to visualize results
class Visualization(object):
    """This class serves visualization purposes of any kind.
    """
    def __init__(self, simulation, workpiece, tool):
        """It is initiated with the Robot class object, a Toolpath class object
        and a Workpiece class object.

        Parameters
        ----------
        parameters : lmfit parameter object
            robot kinematic parameters with which the direct transformation of
            the joint angles is to be executed
        simulation : Simulation class object
            contains the joint angles for the 
        workpiece : Workpiece class object
            contains the workpiece pose
        tool : Tool class object
            contains the tool information
        """
        self.simulation = simulation
        self.workpiece = workpiece
        self.tool = tool

    def plot_pcd_mesh_grid(
        self, objs=[], types=[], colors=[], cs=True, tool=True, pointsize=5, tool_pose=0
        ):
        """Plots and compares the stock and result meshes.

        Parameters
        ----------
        objs : list, optional
            list of meshes and pcds to be plotted, by default []
        types : list, optional
            list of 'solid' or 'wireframe' definitions, by default []
        colors : list, optional
            list of colors as list (3,), by default []
        cs : bool, optional
            whether or not to plot workpiece frame, by default True
        tool : bool, optional
            whether or not to plot tool, by default True
        pointsize : int, optional
            point size of pcd, by default 5

        Raises
        ------
        AttributeError
            raised if unknown type ('solid', 'wireframe') is specified
        """
        
        plot_objs = []
        is_wireframe = False
        for obj, type, color in zip(objs, types, colors):
            if type == 'solid':
                obj.paint_uniform_color(color)
                plot_objs.append(obj)
            elif type == 'wireframe':
                wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(
                    obj
                    )
                wireframe.paint_uniform_color(color)
                plot_objs.append(wireframe)
            elif type == 'grid':
                plot_objs.append(obj)
                is_wireframe = True
            else:
                raise AttributeError('Unknown type parameter for stock')

        if cs:
            plot_objs.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=10))
        if tool:
            tool_geo = o3d.geometry.TriangleMesh.create_cylinder(
                radius=self.tool.diameter/2,
                height=self.tool.flute_length,
                resolution=20,
                create_uv_map=True
                )
            tool_geo.compute_vertex_normals()

            trajectory_rb = [self.simulation.robot_real.rtb_robot.fkine(ja) for ja in self.simulation.joint_angles]
            trajectory_w = [self.workpiece.pose.inv() @ traj for traj in trajectory_rb]

            T = np.copy(trajectory_w[tool_pose].A)
            T[:3, 3] *= 1000
            tool_geo = tool_geo.transform(T)
            tool_geo = tool_geo.translate([0, 0, self.tool.flute_length/2])
            plot_objs.append(tool_geo)

        o3d.visualization.draw_geometries(
            plot_objs,
            width=1920, height=1080, left=0, top=0,
            mesh_show_wireframe=is_wireframe, mesh_show_back_face=True
        )
        
    def plot_robot_and_workpiece(self, toolpath, frames=False, alpha=0.5):
        """Plots the robot, the result mesh and the tool path.

        Parameters
        ----------
        frames : bool, optional
            tool path, by default False
        alpha : float, optional
            transparency of result mesh, by default 0.5
        """
        figure1 = plt.figure()
        env = PyPlot()
        env.launch(fig=figure1, limits=[0, 2, -1, 1, 0, 2])
        
        plt.subplots_adjust(bottom=0.25)
        axpose = plt.axes([0.25, 0.1, 0.65, 0.03])
        pose_slider = Slider(
            ax=axpose,
            label='Pose',
            valmin=1,
            valmax=len(toolpath),
            valstep=1,
            valinit=1,
        )

        def update(val):
            env.add(self.simulation.robot_real.rtb_robot)
            self.simulation.robot_real.rtb_robot.q = self.simulation.joint_angles[val-1]
            env.step()   

        pose_slider.on_changed(update)
        
        env.add(self.simulation.robot_real.rtb_robot, readonly=True, eeframe=True)
        self.simulation.robot_real.rtb_robot.q = self.simulation.joint_angles[0]
        env.step()   

        your_mesh = mesh.Mesh.from_file(self.workpiece.result_file_path)
        your_mesh.points /= 1000
        your_mesh.transform(self.workpiece.pose.A)
        wp = mplot3d.art3d.Poly3DCollection(your_mesh.vectors, edgecolor='k')
        wp.set_alpha(alpha)
        env.ax.add_collection3d(wp)
        if frames:
            plt.sca(env.ax)
            for pose in toolpath:
                trplot(pose.A, block=False, style='rviz', length=0.1, width=0.3)
       
        env.hold()
