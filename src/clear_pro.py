import roboticstoolbox as rtb
from spatialmath import *
from spatialmath.base import *
import numpy as np
from scipy.interpolate import interp1d
from ctypes import cdll, POINTER, c_double, byref
import os
import open3d as o3d
import pyvista as pv
from stl import mesh
import copy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.widgets import Slider
from roboticstoolbox.backends.PyPlot import PyPlot
from scipy.spatial import cKDTree
import lmfit as lf
import configparser
import pickle 



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

    def perturbate_parameters(self, init_parameters, perturbation):
        """takes input parameter object and perturbates the parameters that are
        to be identified

        Parameters
        ----------
        init_parameters : lmfit parameter object
            parameters to be perturbated
        perturbation : degree of perturbation
            normalized to init parameters

        Returns
        -------
        perturbated parameters
            lmfit parameter object
        """
        # perturbate nominal parameters
        perturbated_parameters = copy.deepcopy(init_parameters)
        for _, parameter in perturbated_parameters.items():
            if parameter.vary == True:
                if parameter.value == 0:
                    parameter.value += perturbation*(np.random.randn()-0.5)
                else:
                    parameter.value += perturbation*(np.random.randn()-0.5)*parameter.value

        return perturbated_parameters

    def read_joint_limits(self, file):
        """reads the robot joint limits from file

        Parameters
        ----------
        file : string
            joint limit file path

        Returns
        -------
        numpy array (6, 2)
            array of joint limits
        """
        # read ini file
        config = configparser.ConfigParser()
        config.read(file)
        tmp = config.get('joint limits' , 'lims').split()
        joint_limits = np.array([
            [float(tmp[0]), float(tmp[1])],
            [float(tmp[2]), float(tmp[3])],
            [float(tmp[4]), float(tmp[5])],
            [float(tmp[6]), float(tmp[7])],
            [float(tmp[8]), float(tmp[9])],
            [float(tmp[10]), float(tmp[11])],
        ])
        print('\n')
        print('Joint limits read')

        return joint_limits

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

    def read_toolpath(self, file, nint=10):
        """reads toolpath data from file

        Parameters
        ----------
        file : string
            file path to toolpath data
        nint : int, optional
            number of interpolation points, by default 10

        Returns
        -------
        spatial math SE3 object
            4x4 homogenous transformation matrices of toolpath poses

        Raises
        ------
        AttributeError
            raiescription_
        """
        # read ini file
        config = configparser.ConfigParser()
        config.read(file)
        kind = config.get('kind' , 'kind')
        if 'positions' in config:
            pos_tmp = config.get('positions' , 'positions').split()
            positions = np.asarray(
                [[float(x), float(y), float(z)] 
                for x, y, z in zip(pos_tmp[::3], pos_tmp[1::3], pos_tmp[2::3])]
                )
        if 'orientations' in config:
            ori_tmp = config.get('orientations' , 'orientations').split()
            orientations = np.asarray(
                [[float(A), float(B), float(C)] 
                for A, B, C in zip(ori_tmp[::3], ori_tmp[1::3], ori_tmp[2::3])]
                )
        if 'circle parameters' in config:
            c_tmp = config.get('circle parameters', 'center').split()
            u_tmp = config.get('circle parameters', 'u').split()
            v_tmp = config.get('circle parameters', 'v').split()
            c = np.array([float(c_tmp[0]), float(c_tmp[1]), float(c_tmp[2])])
            u = np.array([float(u_tmp[0]), float(u_tmp[1]), float(u_tmp[2])])
            v = np.array([float(v_tmp[0]), float(v_tmp[1]), float(v_tmp[2])])

        # path interpolation
        if kind == 'linear':
            toolpath_w = [
                SE3(pos*1e-3) * SE3.RPY(ori, order='zyx', unit='deg') 
                for pos, ori in zip(positions, orientations)
                ]   
            print('\n')
            print('Linear toolpath read in')              
        elif kind == 'circle':
            param = np.linspace(0, 1, nint)
            positions = np.asarray([
                c + u*np.cos(2*np.pi/param[-1]*t) + v*np.sin(2*np.pi/param[-1]*t)
                for t in param
                ]          )      
            s = np.linspace(0, 1, orientations.shape[0])
            snew = np.linspace(0, s[-1], nint)
            fA = interp1d(s, orientations[:, 2], kind='linear')
            fB = interp1d(s, orientations[:, 1], kind='linear')
            fC = interp1d(s, orientations[:, 0], kind='linear')
            Anew = fA(snew)
            Bnew = fB(snew)
            Cnew = fC(snew)
            toolpath_w = [
                SE3(pos*1e-3) * SE3.RPY([C, B, A], order='zyx', unit='deg')
                for pos, A, B, C in zip(positions, Anew, Bnew, Cnew)
                ]
            print('\n')
            print('Circular toolpath read in')  
        elif kind == 'spline':
            s1 = np.zeros(1)
            s2 = np.cumsum(np.sqrt(np.diff(
                positions[:, 0])**2+np.diff(positions[:, 1])**2+np.diff(positions[:, 2])**2
                ))
            s = np.hstack((s1, s2))
            snew = np.linspace(0, s[-1], nint)
            fx = interp1d(s, positions[:, 0], kind='cubic')
            fy = interp1d(s, positions[:, 1], kind='cubic')
            fz = interp1d(s, positions[:, 2], kind='cubic')
            xnew = fx(snew)
            ynew = fy(snew)
            znew = fz(snew)
            fA = interp1d(s, orientations[:, 2], kind='linear')
            fB = interp1d(s, orientations[:, 1], kind='linear')
            fC = interp1d(s, orientations[:, 0], kind='linear')
            Anew = fA(snew)
            Bnew = fB(snew)
            Cnew = fC(snew)
            toolpath_w = [
                SE3([x*1e-3, y*1e-3, z*1e-3])* SE3.RPY([C, B, A], order='zyx', unit='deg') 
                for x, y, z, A, B, C in zip(xnew, ynew, znew, Anew, Bnew, Cnew)
                ]
            print('\n')
            print('Spline toolpath read in')  
        else:
            raise AttributeError('Unknown type of toolpath kind')

        return toolpath_w

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
        # change back to main directory
        os.chdir('../../')
        # save result mesh to object
        self.workpiece.result_mesh = o3d.io.read_triangle_mesh(result_path[3:])
        self.workpiece.result_mesh.compute_vertex_normals()
        
        # create pcd
        if is_pcd:
            self.workpiece.result_pcd = self.workpiece.result_mesh.sample_points_poisson_disk(number_of_points=npoints, init_factor=5)

        # create voxel grid
        if is_voxel_grid:
            v_size=round(
                max(self.workpiece.result_mesh.get_max_bound()-self.workpiece.result_mesh.get_min_bound())/
                nvoxels,
                3)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
                self.workpiece.result_mesh, voxel_size=v_size
                )
            self.workpiece.result_voxelgrid = voxel_grid
        else:
            self.workpiece.result_voxelgrid = None

        if is_delete:
            os.remove(result_path[3:])

        if is_print:
            print('\n')
            print('Material removal simulation executed')

        return self.workpiece.result_pcd, self.workpiece.result_mesh, self.workpiece.result_voxelgrid
       
    def get_removed_volume(self, mesh_in, is_plot=False, is_print=False, is_voxel_grid=False):
        """takes mesh_in and substracts it from the stock mesh

        Parameters
        ----------
        mesh_in : open3.geometry.TriangleMesh
            result mesh

        Returns
        -------
        open3d.geometry.TriangleMesh
            removed volume
        """
        mesh_stock_pv = pv.read(self.workpiece.stock_file_path)
        mesh_result_o3d = copy.deepcopy(mesh_in)
        # conversion of open3d mesh to pyvista mesh
        verts = np.asarray(mesh_result_o3d.vertices)
        tmp = np.asarray(mesh_result_o3d.triangles)
        faces = np.hstack(np.insert(tmp, np.arange(0, 3*len(tmp), 3), 3))
        mesh_result_pv = pv.PolyData(verts, faces)
        center = np.asarray(mesh_result_pv.center)

        mesh_stock_pv.translate((-center), inplace=True)
        mesh_result_pv.translate((-center), inplace=True)

        # get removed volume with pyvista boolean difference
        scale = 1.01
        mesh_result_pv.scale((scale, scale, scale), inplace=True)

        # pl = pv.Plotter()
        # _ = pl.add_mesh(mesh_stock_pv, color='r', style='wireframe', line_width=2)
        # _ = pl.add_mesh(mesh_result_pv, color='b', style='wireframe', line_width=3)
        # _ = pl.add_axes_at_origin(xlabel=None, ylabel=None, zlabel=None)
        # pl.camera_position = 'xz'
        # pl.show()

        mesh_removed = mesh_stock_pv.boolean_difference(mesh_result_pv)
        mesh_removed.translate((center), inplace=True)

        # pv.set_plot_theme('default')    # 'ParaView', 'dark', 'document'
        # pl = pv.Plotter()
        # pl.set_background('white')
        # _ = pl.add_mesh(mesh_stock_pv, color='k', style='wireframe', line_width=2)
        # _ = pl.add_mesh(mesh_result_pv, color='g', style='wireframe', line_width=3)
        # _ = pl.add_mesh(mesh_removed, color='tan')
        # _ = pl.add_axes_at_origin(xlabel=None, ylabel=None, zlabel=None)
        # pl.camera_position = 'xz'
        # pl.show()

        # convert pyvista mesh to open3d mesh
        try:
            mask = np.ones(len(mesh_removed.faces), dtype=bool)
            mask[0:-1:4] = False
            mesh_removed_o3d = o3d.geometry.TriangleMesh()
            mesh_removed_o3d.vertices = o3d.utility.Vector3dVector(
                mesh_removed.points
                )
            mesh_removed_o3d.triangles = o3d.utility.Vector3iVector(
                mesh_removed.faces[mask].reshape(-1, 3)
                )
            mesh_removed_o3d.triangle_normals = o3d.utility.Vector3dVector(
                mesh_removed.face_normals
                )
            mesh_removed_o3d.vertex_normals = o3d.utility.Vector3dVector(
                mesh_removed.point_normals
                )
        except:
            print('Boolean difference did not work!')
        # o3d.visualization.draw_geometries([mesh_removed_o3d])

        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_in)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_t)        

        source_points = np.asarray(mesh_removed_o3d.vertices).astype(np.float32)

        # Compute the signed distance for N random points
        dist = scene.compute_distance(source_points)
        verts_in = np.where(dist.numpy()<=1)[0]
        mesh_interface_o3d = o3d.geometry.TriangleMesh()
        mesh_interface_o3d.vertices = o3d.utility.Vector3dVector(
            np.asarray(mesh_removed_o3d.vertices)[verts_in]
            )
        tris_in = []
        for tri in mesh_removed_o3d.triangles:
            if set(tri).issubset(set(verts_in)):
                tris_in.append(tri)
        mesh_interface_o3d.triangles = o3d.utility.Vector3iVector(
            np.asarray(tris_in)
            )
        mesh_interface_o3d.paint_uniform_color([1, 0, 0])

        if is_voxel_grid:
            v_size=round(
                max(mesh_interface_o3d.get_max_bound()-mesh_interface_o3d.get_min_bound())/
                self.workpiece.nvoxels,
                3)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
                mesh_interface_o3d, voxel_size=v_size
                )
            self.workpiece.result_voxelgrid = voxel_grid
        else:
            self.workpiece.result_voxelgrid = None


        if is_plot:
            o3d.visualization.draw_geometries([mesh_removed_o3d, mesh_interface_o3d])

        if is_print:    
            print('\n')
            print('Removed volume calculated')

        return mesh_removed_o3d, self.workpiece.result_voxelgrid

    def get_distance_between_meshes(self, parameters, mesh_target_o3d, is_print=False):
        """runs a material removal simulation and calculates the distance between
        the resulting removed volume and mesh_target_o3d

        Parameters
        ----------
        parameters : lmfit parameter object
            robot kinematic parameters with which a material removal simulation
            is to be executed
        mesh_target_o3d : o3d mesh
            target mesh to which the distance is to be calculated

        Returns
        -------
        float
            sum of distance vector entries
        """
        mesh_result_o3d, _ = self.do_cut(parameters)

        mesh_source_o3d = self.get_removed_volume(mesh_result_o3d)

        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_target_o3d)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_t)        

        source_points = np.asarray(mesh_source_o3d.vertices).astype(np.float32)

        # Compute the signed distance for N random points
        dist = scene.compute_distance(source_points)
        
        if is_print:
            print('\n')
            print('Distance between two meshes calculated')

        return np.sum(dist.numpy()**2)

    def get_distance_between_grids(self, parameters, grid_target_o3d, is_print=False):
        """runs a material removal simulation and calculates the distance between
        the resulting removed volume and mesh_target_o3d

        Parameters
        ----------
        parameters : lmfit parameter object
            robot kinematic parameters with which a material removal simulation
            is to be executed
        mesh_target_o3d : o3d mesh
            target mesh to which the distance is to be calculated

        Returns
        -------
        float
            sum of distance vector entries
        """
        # _, grid_result_o3d = self.do_cut(parameters)        
        
        _, grid_result_o3d = self.do_cut(parameters, is_voxel_grid=True, is_delete=True)
        # _, grid_result_o3d = self.get_removed_volume(mesh_result_o3d, is_voxel_grid=True)

        # o3d.visualization.draw_geometries([grid_result_o3d, grid_target_o3d],mesh_show_wireframe=True,mesh_show_back_face =True)
        
        voxels_target = grid_target_o3d.get_voxels()
        indices = np.stack(list(vx.grid_index for vx in voxels_target))
        voxelarray_target = np.zeros((self.workpiece.nvoxels+1, self.workpiece.nvoxels+1, self.workpiece.nvoxels+1))
        for indexgrid in indices:
            voxelarray_target[indexgrid[0]][indexgrid[1]][indexgrid[2]]=1

        voxels_result = grid_result_o3d.get_voxels()
        indices = np.stack(list(vx.grid_index for vx in voxels_result))
        voxelarray_result = np.zeros((self.workpiece.nvoxels+1, self.workpiece.nvoxels+1, self.workpiece.nvoxels+1))
        for indexgrid in indices:
            voxelarray_result[indexgrid[0]][indexgrid[1]][indexgrid[2]]=1

        # Compute the signed distance for N random points
        # sqerr = np.sum((voxelarray_target - voxelarray_result)**2)
        sqerr = (voxelarray_target - voxelarray_result).flatten()
        
        if is_print:
            print('\n')
            print('Distance between two voxel grids calculated')

        return sqerr

    def identify_parameters_mesh(self, parameters, target_mesh, solution=None):
        """varies the kinematic parameters by comparing the resulting mesh
        with target_mesh

        Parameters
        ----------
        parameters : lmfit parameter object
            contains the ideal kinematic robot parameters (start values)
        target_mesh : o3d mesh
            target mesh of removed volume with real parameters
        solution : lmfit parameter object, optional
            contains the real robot parameters, by default None

        Returns
        -------
        lmfit result object
            contains the resulting parameters and statistical information
        """
        print('\n')
        print('Optimization started')
        minner = lf.Minimizer(
            self.get_distance_between_meshes,
            parameters,
            fcn_args = [target_mesh],
            verbose  = 2,
            method   = 'trf',
            ftol     = 1e-3,
            max_nfev = 1000,
            jac      = '2-point',
            # iter_cb  = iter_fcn
        )    
        result = minner.minimize(
            method = 'least_squares'
        )
        print('\n')
        print('Parameters opt / real:')
        print('-----------------------------')
        for (key1, parameter1), (key2, parameter2) in zip(
            result.params.items(), solution.items()):
            if parameter1.vary == True:
                print('{}: {:.3f} / {}: {:.3f}'.format(
                    key1, parameter1.value, key2, parameter2.value
                    )
                    )
        print('\n')
        return result

    def identify_parameters_grid(self, parameters, target_grid, solution=None):
        """varies the kinematic parameters by comparing the resulting voxel grid
        with the target voxelgrid

        Parameters
        ----------
        parameters : lmfit parameter object
            contains the ideal kinematic robot parameters (start values)
        target_mesh : o3d mesh
            target mesh of removed volume with real parameters
        solution : lmfit parameter object, optional
            contains the real robot parameters, by default None

        Returns
        -------
        lmfit result object
            contains the resulting parameters and statistical information
        """

        def iteration_callback(params, iter, resid, voxel_grid):
            for key, parameter in params.items():
                if parameter.vary == True:
                    print(key+': '+str(parameter.value))
            print('iter: '+str(iter))
            print('resid: '+str(resid))

        print('\n')
        print('Optimization started')
        minner = lf.Minimizer(
            self.get_distance_between_grids,
            parameters,
            fcn_args = [target_grid],
            verbose  = 2,
            method   = 'trf',
            ftol     = 1e-5,
            max_nfev = 1000,
            jac      = '2-point',
            # diff_step = 1e-1,
            iter_cb  = iteration_callback
        )    
        result = minner.minimize(
            method = 'least_squares'
        )
        print('\n')
        print('Parameters opt / real:')
        print('-----------------------------')
        for (key1, parameter1), (key2, parameter2) in zip(
            result.params.items(), solution.items()):
            if parameter1.vary == True:
                print('{}: {:.3f} / {}: {:.3f}'.format(
                    key1, parameter1.value, key2, parameter2.value
                    )
                    )
        print('\n')
        return result

    def grid_search_grid(
        self, parameters, target, nkeep=5, nvals=3, is_print=False
        ):
        """Performs a brute force grid search to calculate the objective
        function contour surface and get candidates for subsequent local
        minimization using a least squares optimization.

        Parameters
        ----------
        parameters : lmfit parameter object
            contains the fixed and varying robot model parameters
        target : open3d voxel grid
            target voxel grid to compare the actual workpiece geometry
            against
        nkeep : int, optional
            number of candidates to keep for further local
            minimization, by default 5
        nvals : int, optional
            number of steps for parameter variation, by default 3
        is_print : bool, optional
            whether or not to print out result information, by default False

        Returns
        -------
        lmfit minimization result objects
            grid search results and local minimization results
        """
        fitter = lf.Minimizer(
            self.get_distance_between_grids,
            parameters,
            fcn_args = [target],
            )
        result_brute = fitter.minimize(
            method='brute',
            Ns=nvals,
            keep=nkeep
            )

        best_result = copy.deepcopy(result_brute)
        for candidate in result_brute.candidates:
            trial = fitter.minimize(
                method='least_squares',
                params=candidate.params,
                diff_step=1e-1,
                verbose=2
                )
            if trial.chisqr < best_result.chisqr:
                best_result = trial

        if is_print:
            print('\nbest solution (brute):')
            print('--------------')
            print(result_brute.brute_x0)
            print('\nminimum value (brute):')
            print('--------------')
            print(result_brute.brute_fval)
            print('\ngrid values (brute):')
            print('------------')
            print(result_brute.brute_grid)
            print('\ngrid function values (brute):')
            print('---------------------')
            print(result_brute.brute_Jout)

            params_vary = []
            for _, parameter in best_result.params.items():
                if parameter.vary == True:
                    params_vary.append(parameter.value)
            print('\nbest solution (least squares):')
            print('---------------------')
            print(params_vary)
            print('\nminimum solution (least squares):')
            print('---------------------')
            print(best_result.chisqr)

        filename_brute = 'docs/latex/data/grid_search_nvals-'+str(nvals)+'_brute.dictionary'
        with open(filename_brute, 'wb') as result_file:
            pickle.dump(result_brute, result_file)
        filename_best = 'docs/latex/data/grid_search_nvals-'+str(nvals)+'_best.dictionary'
        with open(filename_best, 'wb') as result_file:
            pickle.dump(best_result, result_file)

        return result_brute, best_result



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
