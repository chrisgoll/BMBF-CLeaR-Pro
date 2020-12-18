import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class kinematic:
    """[summary]
    """
    def __init__(self, par):
        """initializes the kinematics by specifying the d-h-parameters

        :param par: denavit-hartenberg-parameters
        :type par: dictionary
        """
        # Denavit-Hartenberg-Parameter
        self.theta = np.array([ 0,          0,          0,       -90,        180,       180,   0,       -90])*np.pi/180
        self.d =     np.array([ 0, par['l11'],          0,         0,          0, par['l4'],   0, par['l6']])
        self.a =     np.array([ 0,          0, par['l12'], par['l2'], -par['l3'],         0,   0,         0])
        self.alpha = np.array([ 0,          0,        -90,         0,         90,        90, -90,         0])*np.pi/180

        print('\n')
        print('robot kinematics initialized')


    def linear_interpolation(self, start_xyz, end_xyz, start_abg, end_abg, npoints):
        x = np.linspace(start_xyz[0],end_xyz[0], npoints)
        y = np.linspace(start_xyz[1],end_xyz[1], npoints)
        z = np.linspace(start_xyz[2],end_xyz[2], npoints)
        alpha = np.linspace(start_abg[0],end_abg[0], npoints)*np.pi/180
        beta  = np.linspace(start_abg[1],end_abg[1], npoints)*np.pi/180
        gamma = np.linspace(start_abg[2],end_abg[2], npoints)*np.pi/180

        print('\n')
        print('linear motion interpolated')

        return np.vstack((x, y, z, alpha, beta, gamma)).T


    def inverse(self, tcp):
        x = tcp[:, 0]
        y = tcp[:, 1]
        z = tcp[:, 2]
        alpha = tcp[:, 3]
        beta  = tcp[:, 4]
        gamma = tcp[:, 5]

        q = []
        for cnt in range(0, len(alpha)):
                
            R = self.R(alpha[cnt], beta[cnt], gamma[cnt])
            n = R[:, 2][:, np.newaxis]
            p = np.array([[x[cnt]],
                          [y[cnt]],
                          [z[cnt]]])
            
            p04 = p - self.d[-1]*n - np.array([[0],[0],[self.d[1]]])
            
            # q1
            q1 = np.arctan2(p04[1], p04[0])
            
            # q3
            p14 = p04 - self.a[2]*np.array([[np.cos(q1)],
                                            [np.sin(q1)],
                                                          [0]])
            l34 = np.sqrt(self.a[4]**2+self.d[5]**2)
            phi1 = np.arccos((self.a[3]**2+l34**2-np.linalg.norm(p14)**2)/
                              (2*self.a[3]*l34))
            phi2 = np.arccos(self.d[5]/l34)
            q3 = 3/2*np.pi-phi1+phi2
            
            # q2
            p14_1 = np.array([[ np.cos(q1), np.sin(q1),  0],
                              [               0,          0,      -1],
                              [-np.sin(q1), np.cos(q1),  0]]).dot(p14).astype(float)
            beta1 = np.arctan2(-p14_1[1],p14_1[0])
            beta2 = np.arccos((self.a[3]**2+np.linalg.norm(p14)**2-l34**2)/
                              (2*self.a[3]*np.linalg.norm(p14)))
            q2 = -(beta1+beta2)
            
            # q5
            A_01 = np.array([[ np.cos(q1-self.theta[2]),  0,-np.sin(q1-self.theta[2])],
                             [ np.sin(q1-self.theta[2]),  0, np.cos(q1-self.theta[2])],
                             [                             0, -1,                             0]])
            A_12 = np.array([[ np.sin(q2-self.theta[3]), np.cos(q2-self.theta[3]),  0],
                             [-np.cos(q2-self.theta[3]), np.sin(q2-self.theta[3]),  0],
                             [                             0,                             0,  1]])
            A_23 = np.array([[-np.cos(q3-self.theta[4]), 0, -np.sin(q3-self.theta[4])],
                             [-np.sin(q3-self.theta[4]), 0,  np.cos(q3-self.theta[4])],
                             [                             0, 1,                              0]])
            x3 = A_01.dot(A_12).dot(A_23).astype(float)[0:3,0]
            y3 = A_01.dot(A_12).dot(A_23).astype(float)[0:3,1]
            z3 = A_01.dot(A_12).dot(A_23).astype(float)[0:3,2]
            q5norm = np.arccos(np.dot(z3.T,n))
            
            d0 = np.cross(z3,n.T)/np.linalg.norm(np.cross(z3,n.T))
            eta = np.absolute(np.arccos(np.dot(y3.T,d0.T)))
            if eta < np.pi/2:
                q5 = q5norm
            else:
                q5 = -q5norm
            
            # q4
            if eta < np.pi/2:
                c0 = np.cross(z3,n.T)/np.linalg.norm(np.cross(z3,n.T))
            else:
                c0 = np.cross(n.T,z3)/np.linalg.norm(np.cross(n.T,z3))
            dq4 = np.arccos(np.dot(y3.T,c0.T))
            chi = np.arccos(np.dot(x3.T,c0.T))
            if chi < np.pi/2:
                dq4 = -np.linalg.norm(dq4)
            else:
                dq4 = np.linalg.norm(dq4)
                
            q4 = np.pi + dq4
            
            # q6
            A_34 = np.array([[-np.cos(q4-self.theta[5]), 0, -np.sin(q4-self.theta[5])],
                             [-np.sin(q4-self.theta[5]), 0,  np.cos(q4-self.theta[5])],
                             [                             0, 1,                              0]])
            A_45 = np.array([[np.cos(q5-self.theta[6]),  0, -np.sin(q5-self.theta[6])],
                             [np.sin(q5-self.theta[6]),  0,  np.cos(q5-self.theta[6])],
                             [                            0, -1,                              0]])
            x5 = A_01.dot(A_12).dot(A_23).dot(A_34).dot(A_45).astype(float)[0:3,0]
            y5 = A_01.dot(A_12).dot(A_23).dot(A_34).dot(A_45).astype(float)[0:3,1]
            l = R[0:3,0]
            q6norm = np.arccos(np.dot(x5.T,l.T))
            delta = np.arccos(np.dot(y5.T,l.T))
            if delta <= np.pi/2:
                q6 = np.absolute(q6norm)
            else:
                q6 = -np.absolute(q6norm)

            q_cnt = [
                0,
                0,
                q1[0]-self.theta[2],
                q2[0]-self.theta[3],
                q3[0]-self.theta[4],
                q4-self.theta[5],
                q5[0]-self.theta[6],
                q6-self.theta[7]]

            q.append(q_cnt)

        return np.asarray(q)


    def direct(self, angles):
        tcp_xyz_abg = []
        # configurations
        for cnt1 in range(angles.shape[0]):
            frame = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
            # joints
            for cnt2 in range(0,8): 
                frame = frame.dot(self.frames(self.theta[cnt2] + angles[cnt1, cnt2], self.d[cnt2], self.a[cnt2] , self.alpha[cnt2]))
            xyz = frame[0:3,3]
            abg = np.asarray(self.euler_angles_zyx(frame))
            tcp_xyz_abg.append([xyz[0], xyz[1], xyz[2], abg[0], abg[1], abg[2]])

        return np.asarray(tcp_xyz_abg)


    def plot_tcp_3d(self, tcp_list):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for tcp in tcp_list:
            ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], lw=2)
        ax.set_xlabel('x in mm')
        ax.set_ylabel('y in mm')
        ax.set_zlabel('z in mm')
        plt.show()


    def plot_joint_angles(self, angles):
        # plt.xkcd()
        _, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
            3, 2, sharex=True, sharey=False
            )
        ax1.plot(np.degrees(angles[:, 2]))
        ax1.set_ylabel('q1')

        ax2.plot(np.degrees(angles[:, 3]))
        ax2.set_ylabel('q2')

        ax3.plot(np.degrees(angles[:, 4]))
        ax3.set_ylabel('q3')

        ax4.plot(np.degrees(angles[:, 5]))
        ax4.set_ylabel('q4')
        
        ax5.plot(np.degrees(angles[:, 6]))
        ax5.set_ylabel('q5')
        ax5.set_xlabel('time')

        ax6.plot(np.degrees(angles[:, 7]))
        ax6.set_ylabel('q6')
        ax6.set_xlabel('time')

        plt.show()


    def R(self, alpha, beta, gamma):

        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)
        cg = np.cos(gamma)
        sg = np.sin(gamma)
        # Z-Y-X Euler angles (alpha, beta, gamma)
        R = np.array([ [ca*cb, -sa*cg-ca*sb*sg,  sa*sg-ca*sb*cg],
                       [sa*cb,  ca*cg-sa*sb*sg, -ca*sg-sa*sb*cg],
                       [   sb,           cb*sg,           cb*cg] ])
        return R


    def euler_angles_zyx(self, T):
        """returns the angles alpha, beta and gamma by the consecutive axis
        z, y and x for the frame T

        :param T: frame
        :type T: np.ndarray[4,4]
        :return: angles alpha, beta, gamma in rad
        :rtype: numbers
        """
        # z-y-x euler angles alpha, beta, gamma
        beta  = np.arctan2(-T[2, 0], np.sqrt(T[0, 0]**2 + T[1, 0]**2))
        alpha = np.arctan2(T[1, 0]/np.cos(beta), T[0, 0]/np.cos(beta))
        gamma = np.arctan2(T[2, 1]/np.cos(beta), T[2, 2]/np.cos(beta))
        return alpha, beta, gamma


    def frames(self, theta, d, a, alpha):
        T = np.array([ [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)], 
                       [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                       [            0,                np.sin(alpha),                np.cos(alpha),               d],
                       [            0,                            0,                            0,               1] ])
        return T
