#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
HW 1: AutoCalib

Author(s):
Ajith Kumar Jayamoorthy (ajayamoorthy@wpi.edu)
MS in Robotics Engineering,
Worcester Polytechnic Institute
"""
from statistics import mean
from scipy import optimize as optim
import numpy as np
import cv2
import glob
import os
import pry

# Add any python libraries here
class Calib:
    def __init__(self):
        pass

    def splitname(self, Images):
        names = []
        for image in Images:
            name  = image.split("/")[-1]
            names.append(name)
        return names

    def Homography(self, Images, Corner_x, Corner_y, Img_names):
        x = [i for i in range(Corner_x)]
        y = [i for i in range(Corner_y)]
        xx, yy          = np.meshgrid(x,y)
        points          = np.vstack((xx.flatten(),yy.flatten())).astype(np.float32).T
        scaled_points   = points*21.5 # Each point is 21.5 mm away from each other, so we are accordingly measuring it.

        pointsOnImage   = [] # 2d points in image planes
        homography      = []
        criteria        = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        i=0
        for image,name in zip(Images,Img_names):
            img         = cv2.imread(image)
            img_gray    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Finding the ChessBoard Corners
            retval, corners = cv2.findChessboardCorners(img_gray,(Corner_x,Corner_y))

            if retval == True:
                corners = corners.reshape(corners.shape[0],corners.shape[2])
                corners2 = cv2.cornerSubPix(img_gray,corners, (11,11), (-1,-1), criteria)
                pointsOnImage.append(corners2)

                H,_ = cv2.findHomography(scaled_points[:30],corners[:30])
                homography.append(H)

                cv2.drawChessboardCorners(img, (Corner_x,Corner_y), corners2, retval)
                cv2.imwrite(os.path.join( 'Output/Homography_output/',name), img)
                cv2.imshow('img', img)                
                cv2.waitKey(0)

        return homography,pointsOnImage,scaled_points

    def get_v(self,i,j,H):
        v = np.array([ H[0,i]*H[0,j], 
                       H[0,i]*H[1,j] + H[1,i]*H[0,j],
                       H[1,i]*H[1,j],
                       H[2,i]*H[0,j] + H[0,i]*H[2,j],
                       H[2,i]*H[1,j] + H[1,i]*H[2,j],
                       H[2,i]*H[2,j]]).reshape(1,6)
        return v

    def IntrinsicParams(self, homographies):
        # Let U = inv(A).T * inv(A)
        V = np.zeros(6).reshape(1,6)
        for H in homographies:
            v12     = self.get_v(0,1,H)
            vdif    = self.get_v(0,0,H)-self.get_v(1,1,H)
            v_vec   = np.vstack((v12,vdif))
            V       = np.vstack((V,v_vec))
        V = V[1:]

        '''
            Source: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf
            The solution to Vb = 0 is well known as the eigenvector of (V.T*V) associated with the smallest
            eigenvalue (equivalently, the right singular vector of V associated with the smallest singular value).
        '''
        # val, vec    = np.linalg.eig(V) # Used for computing eigen vectors and values for sqaure matrix
        u,s,vh      = np.linalg.svd(V) # For calculating eigen vector and values for rectangular matrix
        b           = vh[np.argmin(s)]

        B11 = b[0]
        B12 = b[1]
        B22 = b[2]
        B13 = b[3]
        B23 = b[4]
        B33 = b[5]

        v0      = (B12*B13 - B11*B23)/(B11*B22 - B12**2)
        lamda   = B33 - (B13**2 + v0*(B12*B13 - B11*B23))/B11
        alpha   = np.sqrt(lamda/B11)
        beta    = np.sqrt(lamda*B11 /(B11*B22 - B12**2))
        gamma   = -B12*(alpha**2)*beta/lamda
        u0      = gamma*v0/beta -B13*(alpha**2)/lamda

        K = np.array([[alpha, gamma, u0],
                      [    0,  beta, v0],
                      [    0,     0,  1]])
                    
        return K

    def ExtrinsicParams(self, K, homography):
        R_list = []
        t_list = []
        for H in homography:
            K_inv   = np.linalg.pinv(K)
            # Calculating lamda as follows: λ = ( 1 / ||K_inv*h1|| + 1 / ||K_inv−h2|| ) / 2
            lamda   = mean([1.0/np.linalg.norm(K_inv@H[:,0]),1.0/np.linalg.norm(K_inv@H[:,1])])
            r1      = lamda*(K_inv@H[:,0])
            r2      = lamda*(K_inv@H[:,1])
            r3      = np.cross(r1,r2)
            R       = np.array([r1,r2,r3]).T
            t       = lamda*(K_inv@H[:,2])
            R_list.append(R)
            t_list.append(t)
        
        return R_list,t_list

    def ReprojectionError(self, flag, K, R_list, t_list, pointsOnImage, scaled_points, k1=0, k2=0):
        k1      = k1
        k2      = k2
        u0      = K[0,2]
        v0      = K[1,2]
        mean_err_list = []
        reproj_points = []

        for R,t,img_points in zip(R_list,t_list,pointsOnImage):
            E           = np.hstack((R,t.reshape(-1,1)))  # Extrinsic Matrix
            P           = K@E                             # Projection Matrix
            err_list    = []
            reproj_pts  = np.array([0,0]).reshape(1,2)
            for img_pt, scaled_pt in zip(img_points,scaled_points):
                if flag == 0:
                    img_pt      = np.array([img_pt[0], img_pt[1], 1]).reshape(3,1) 
                    
                    # Converting scaled_points as projection on the image
                    model_pts   = np.array([scaled_pt[0],scaled_pt[1], 0, 1]).reshape(4,1)
                    proj_pts    = P @ model_pts
                    proj_pts    = proj_pts/proj_pts[2] # Projected point normalizing with respect to the Z axis

                    err         = np.linalg.norm(img_pt-proj_pts,ord=2)
                    err_list.append(err)
                else:
                    img_pt      = np.array(img_pt).reshape(2,1)
                    # Converting scaled_points as project on the image
                    model_pts   = np.array([scaled_pt[0],scaled_pt[1], 0, 1]).reshape(4,1)
                    proj_pts    = E @ model_pts
                    proj_pts    = proj_pts/proj_pts[2] # Projected point normalizing with respect to the Z axis
                    x,y         = proj_pts[0], proj_pts[1]

                    uv          = P @ model_pts 
                    uv          = uv/uv[2]
                    u,v         = uv[0],uv[1]

                    u_cap       = u + (u-u0) * [k1*(x**2 + y**2) + k2*(x**2 + y**2)**2]
                    v_cap       = v + (v-v0) * [k1*(x**2 + y**2) + k2*(x**2 + y**2)**2]
                    uv_cap      = np.array([u_cap.flatten(),v_cap.flatten()])
                    err         = np.linalg.norm(img_pt-uv_cap,ord=2)
                    reproj_pts  = np.vstack((reproj_pts,uv_cap.T))
                    err_list.append(err)

            reproj_pts = reproj_pts[1:]
            reproj_points.append(reproj_pts)    
            mean_err_list.append(mean(err_list))

        if flag==0:
            return mean_err_list
        else:
            return mean_err_list,reproj_points

    
    def optFunc(self,init_params,pointsOnImage,scaled_points, homography):
        K       = np.zeros((3,3))
        K[0,0]  = init_params[0]
        K[1,1]  = init_params[1]
        K[0,1]  = init_params[2]
        K[0,2]  = init_params[3]
        K[1,2]  = init_params[4]
        K[2,2]  = 1
        k1      = init_params[5]
        k2      = init_params[6]
        u0      = init_params[3]
        v0      = init_params[4]

        R_list,t_list   = self.ExtrinsicParams(K,homography)    
        residualList    = []

        for R,t,img_points in zip(R_list,t_list,pointsOnImage):
            E = np.hstack((R,t.reshape(-1,1)))  # Extrinsic Matrix
            P = K@E                             # Projection Matrix

            for img_pt, scaled_pt in zip(img_points,scaled_points):               
                # Converting scaled_points as project on the image
                model_pts   = np.array([scaled_pt[0],scaled_pt[1], 0, 1]).reshape(4,1)
                proj_pts    = E @ model_pts
                proj_pts    = proj_pts/proj_pts[2] # Projected point normalizing with respect to the Z axis
                x,y         = proj_pts[0], proj_pts[1]

                uv          = P @ model_pts 
                uv          = uv/uv[2]
                u,v         = uv[0],uv[1]

                u_cap       = u + (u-u0) * [k1*(x**2 + y**2) + k2*(x**2 + y**2)**2]
                v_cap       = v + (v-v0) * [k1*(x**2 + y**2) + k2*(x**2 + y**2)**2]
                residualList.append(img_pt[0]-u_cap)
                residualList.append(img_pt[1]-v_cap)

        residualList = np.array(residualList,dtype=float).flatten()
        return residualList

    def optimize(self, K, pointsOnImage, scaled_points, homography):
        # Approximate distortion k
        k_init      = np.array([0 , 0]).T 
        k1          = k_init[0]
        k2          = k_init[1]        
        init_params = np.array([K[0,0], K[1,1], K[0,1], K[0,2], K[1,2], k1, k2])
        
        optim_params    = optim.least_squares(fun=self.optFunc, x0=init_params, method='lm', args=(pointsOnImage,scaled_points, homography))
        final_params    = optim_params.x

        K_opt       = np.zeros((3,3))
        K_opt[0,0]  = final_params[0]
        K_opt[1,1]  = final_params[1]
        K_opt[0,1]  = final_params[2]
        K_opt[0,2]  = final_params[3]
        K_opt[1,2]  = final_params[4]
        K_opt[2,2]  = 1
        k1_opt      = final_params[5]
        k2_opt      = final_params[6]
        return K_opt, k1_opt, k2_opt

    def rectifyImg(self,pointsOnImage,opt_pts,Images,Img_names):
        for image, img_pts, optim_pts, name in zip(Images, pointsOnImage, opt_pts, Img_names):
            img         = cv2.imread(image)
            H,_         = cv2.findHomography(img_pts,optim_pts)
            img_warp    = cv2.warpPerspective(img,H,(img.shape[1],img.shape[0]))
            cv2.imwrite('Output/Warped_output/'+name, img_warp)

    def visualize(self,pointsOnImage, opt_pts, Images, Img_names):
        for image, img_pts, optim_pts, name in zip(Images, pointsOnImage, opt_pts, Img_names):
            img = cv2.imread(image)
            for im_pt, opt_pt in zip(img_pts, optim_pts):
                [x, y] = np.int64(im_pt)
                [x_correct, y_correct] = np.int64(opt_pt)
                cv2.circle(img, (x, y), 5, (100, 100, 255),thickness=2) # Original Orange
                cv2.circle(img, (x_correct, y_correct), 5, (255, 0, 0), thickness=2) # Corrected Blue
            cv2.imwrite('Output/Visualize_output/'+name, img)


def main():
    calibration = Calib()
    path        = 'Data/Calibration_Imgs/*.jpg'
    Images      = sorted(glob.glob(path))
    Img_names   = calibration.splitname(Images)    
    
    Corner_x = 9
    Corner_y = 6
    flag     = 0 # flag=0 Reprojection error without distortion / flag=1 Reprojection error with distortion
    
    H,pointsOnImage,scaled_points   = calibration.Homography(Images, Corner_x, Corner_y, Img_names) 
    K                               = calibration.IntrinsicParams(H)
    R_list,t_list                   = calibration.ExtrinsicParams(K, H)
    reproj_err_list                 = calibration.ReprojectionError(flag, K, R_list, t_list, pointsOnImage, scaled_points)
    K_final,k1,k2                   = calibration.optimize(K, pointsOnImage, scaled_points, H)
    R_list_opt,t_list_opt           = calibration.ExtrinsicParams(K_final, H)
    reproj_err_list_opt,opt_pts     = calibration.ReprojectionError(1, K_final, R_list_opt, t_list_opt, pointsOnImage, scaled_points, k1, k2)
    
    calibration.rectifyImg(pointsOnImage,opt_pts,Images,Img_names)
    warped_path     = 'Output/Warped_output/*.jpg'
    Warped_Images   = sorted(glob.glob(warped_path))
    calibration.visualize(pointsOnImage, opt_pts, Warped_Images, Img_names)

    print("Mean Reprojection error before optimization      :"   , np.mean(reproj_err_list))
    print("Initial Calibration matrix is: \n",K)
    print("\nMean Reprojection error after optimization     :"   , np.mean(reproj_err_list_opt))
    print("Final Calibration matrix is: \n",K_final)
    print("\nDistortion coefficients after optimization     :"   , k1, k2)
    
if __name__ == "__main__":
    main()