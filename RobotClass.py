import numpy as np


class kinematic:
    """[summary]
    """
    def __init__(self, par):
        """initializes the kinematics by specifying the d-h-parameters

        :param par: denavit-hartenberg-parameters
        :type par: dictionary
        """
        # Denavit-Hartenberg-Parameter
        self.theta = np.array([ 0, 0, 0, -90, 180, 180, 0, -90])*np.pi/180
        self.d =     np.array([ 0, par['l11'], 0, 0, 0,  par['l4'], 0, par['l6']])
        self.a =     np.array([ 0, 0, par['l12'], par['l2'], -par['l3'], 0, 0, 0])
        self.alpha = np.array([ 0, 0, -90, 0, 90, 90, -90, 0])*np.pi/180

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

        q1 = np.zeros((len(alpha),1))
        q2 = np.zeros((len(alpha),1))
        q3 = np.zeros((len(alpha),1))
        q4 = np.zeros((len(alpha),1))
        q5 = np.zeros((len(alpha),1))
        q6 = np.zeros((len(alpha),1))
        
        for cnt in range(0, len(alpha)):
                
            R = self.R(alpha[cnt], beta[cnt], gamma[cnt])
            n = R[:, 2][:, np.newaxis]
            p = np.array([[x[cnt]],
                          [y[cnt]],
                          [z[cnt]]])
            
            p04 = p - self.d[-1]*n - np.array([[0],[0],[self.d[1]]])
            
            # q1
            q1[cnt] = np.arctan2(p04[1], p04[0])
            
            # q3
            p14 = p04 - self.a[2]*np.array([[np.cos(q1[cnt])],
                                            [np.sin(q1[cnt])],
                                                          [0]])
            l34 = np.sqrt(self.a[4]**2+self.d[5]**2)
            phi1 = np.arccos((self.a[3]**2+l34**2-np.linalg.norm(p14)**2)/
                              (2*self.a[3]*l34))
            phi2 = np.arccos(self.d[5]/l34)
            q3[cnt] = 3/2*np.pi-phi1+phi2
            
            # q2
            p14_1 = np.array([[ np.cos(q1[cnt]), np.sin(q1[cnt]),  0],
                              [               0,          0,      -1],
                              [-np.sin(q1[cnt]), np.cos(q1[cnt]),  0]]).dot(p14).astype(float)
            beta1 = np.arctan2(-p14_1[1],p14_1[0])
            beta2 = np.arccos((self.a[3]**2+np.linalg.norm(p14)**2-l34**2)/
                              (2*self.a[3]*np.linalg.norm(p14)))
            q2[cnt] = -(beta1+beta2)
            
            # q5
            A_01 = np.array([[ np.cos(q1[cnt]-self.theta[2]),  0,-np.sin(q1[cnt]-self.theta[2])],
                             [ np.sin(q1[cnt]-self.theta[2]),  0, np.cos(q1[cnt]-self.theta[2])],
                             [                             0, -1,                             0]])
            A_12 = np.array([[ np.sin(q2[cnt]-self.theta[3]), np.cos(q2[cnt]-self.theta[3]),  0],
                             [-np.cos(q2[cnt]-self.theta[3]), np.sin(q2[cnt]-self.theta[3]),  0],
                             [                             0,                             0,  1]])
            A_23 = np.array([[-np.cos(q3[cnt]-self.theta[4]), 0, -np.sin(q3[cnt]-self.theta[4])],
                             [-np.sin(q3[cnt]-self.theta[4]), 0,  np.cos(q3[cnt]-self.theta[4])],
                             [                             0, 1,                              0]])
            x3 = A_01.dot(A_12).dot(A_23).astype(float)[0:3,0]
            y3 = A_01.dot(A_12).dot(A_23).astype(float)[0:3,1]
            z3 = A_01.dot(A_12).dot(A_23).astype(float)[0:3,2]
            q5norm = np.arccos(np.dot(z3.T,n))
            
            d0 = np.cross(z3,n.T)/np.linalg.norm(np.cross(z3,n.T))
            eta = np.absolute(np.arccos(np.dot(y3.T,d0.T)))
            if eta < np.pi/2:
                q5[cnt] = q5norm
            else:
                q5[cnt] = -q5norm
            
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
                
            q4[cnt] = np.pi + dq4
            
            # q6
            A_34 = np.array([[-np.cos(q4[cnt]-self.theta[5]), 0, -np.sin(q4[cnt]-self.theta[5])],
                             [-np.sin(q4[cnt]-self.theta[5]), 0,  np.cos(q4[cnt]-self.theta[5])],
                             [                             0, 1,                              0]])
            A_45 = np.array([[np.cos(q5[cnt]-self.theta[6]),  0, -np.sin(q5[cnt]-self.theta[6])],
                             [np.sin(q5[cnt]-self.theta[6]),  0,  np.cos(q5[cnt]-self.theta[6])],
                             [                            0, -1,                              0]])
            x5 = A_01.dot(A_12).dot(A_23).dot(A_34).dot(A_45).astype(float)[0:3,0]
            y5 = A_01.dot(A_12).dot(A_23).dot(A_34).dot(A_45).astype(float)[0:3,1]
            l = R[0:3,0]
            q6norm = np.arccos(np.dot(x5.T,l.T))
            delta = np.arccos(np.dot(y5.T,l.T))            
            if delta <= np.pi/2:
                q6[cnt] = np.absolute(q6norm)
            else:
                q6[cnt] = -np.absolute(q6norm)

        return [q1, q2, q3, q4, q5, q6]


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
