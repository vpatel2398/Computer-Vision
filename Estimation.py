##### !!!!!DO NOT CHANGE OR ADD ANY IMPORTS WITHOUT PERMISSION!!!!!
##### IMPORTS START #####
# Imported modules, do not change or add anything
import imageio, sys, os, numpy as np, zipfile, math
from tqdm import tqdm
from skimage.transform import warp
from skimage.transform import resize
import torch
import torch.nn as nn
try:
    from google.colab import drive
    os.system('pip install -U opencv-contrib-python==4.4.0.44')
    import cv2
    print(f'\ncv2 version: {cv2.__version__}\n')
    sift_testing_blah = cv2.SIFT_create(nOctaveLayers = 3, contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6)
except:
    import cv2
    sift_testing_blah = cv2.SIFT_create(nOctaveLayers = 3, contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6)
##### IMPORTS END #####



# Class you will be turning in
# DO NOT put any viewing functions in the class, do this in main.py
dtype = torch.float
device = torch.device("cpu")
class estimate_transforms:
    # Constructor
    def __init__(self, img_path_reg, img_path_warped):
        # Open images
        self.img_reg = imageio.imread(img_path_reg)
        self.img_warped = imageio.imread(img_path_warped)


    
    ##### HELPERS SECTIONS #####
    ## Place any additional helpers here
    ## Dont need to use these, can create whatever you like

    ## QR Decomp
    def qr_solve(self, H, b):
        # H is n by n matrix, b is a n by 1 vector
        # returns a n by 1 vector as a solution
        Q, R = np.linalg.qr(H, 'reduced')
        b_dash = Q.transpose(1,0) @ b
        Np = b_dash.shape[0]
        del_p = np.zeros((Np, 1))
        for i in range(Np-1, -1, -1): # work from the last row of R
            sum_r_p = [0]
            for j in range(i+1, Np, 1) :
                sum_r_p += R[i, j]*del_p[j]
            if (R[i,i] != 0.0):
                del_p[i] = (b_dash[i]- sum_r_p)/R[i,i]
    
        return(del_p)
    
    # this function will help detect and matching the points of two images
    def detect_and_matching_point(self, image_1, image_2):
        
        sift = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6) 
        kp_1, desc_1 = sift.detectAndCompute(image_1, None)  #getting te key points and description of each image
        kp_2, desc_2 = sift.detectAndCompute(image_2, None)
        
        #Match SIFT
        #match the descriptions of both images using cv2
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(desc_1, desc_2)
        matches = sorted(matches, key=lambda x:x.distance)
        
        #extract points
        #extract the points and store them in the list 
        points_reg = []  # empty list for points for first image
        points_warped = [] # empty list for points for second image
        for i in range(len(matches)):
            points_reg.append([kp_1[matches[i].queryIdx].pt[0], kp_1[matches[i].queryIdx].pt[1]]) #adding points to the lists
            points_warped.append([kp_2[matches[i].trainIdx].pt[0], kp_2[matches[i].trainIdx].pt[1]])
            
        points_reg = np.array(points_reg) # converting the list to array
        points_warped = np.array(points_warped)
        
        return (points_reg, points_warped, kp_1, kp_2, matches)

    ## Get SIFT Feature Point Pairs
    def get_sift_points(self, num_matches):
        img_1 = self.img_reg  # reading the main image
        img_2 = self.img_warped  # reading the second/warped image
        
        # getting points for both images from detect_and_matching_point function
        points_reg, points_warped, kp_1, kp_2, matches = self.detect_and_matching_point(img_1, img_2) 
        points_reg = points_reg[:num_matches,:]
        points_warped = points_warped[:num_matches,:]
        
        return (points_reg, points_warped)

    
    def ransac_N(self, m, u, p) :
    #m: minimum number of data units needed to estimate
        #u: what fraction of data are inliers
        #p: how confident we want to be in our estimate
    
        N = np.log(1 - p)/np.log(1 - np.power(u, m)) 
        return(np.int(N))
    
    def detect_intliers (self, x, x_dash, Transform, acceptable_error = 2) :
        # input: two 2D points sets, 3 by N arrays of homogeneous representation of the points
        # Transform: estimated transform to use to detect inliers.
        # acceptable_error - amount of average pixel error that is acceptable for inliers
        # output is an 1D array of size equal to the number of points with 0 or 1, 
        #        with 1 indicating the corresponding point is an inlier
        
        X_t = Transform @ x
        X_t = np.divide(X_t, X_t [2,:]) # normalized homogenous coordinates
        X_dash = np.divide(x_dash, x_dash [2,:]) # normalized homogenous coordinates
           
        error =  (x_dash - X_t)
        residual_error = np.sum(np.power(error, 2), axis=0)
        inliers  = np.where(residual_error < (acceptable_error*acceptable_error), 1, 0)
        
        return(inliers)


    ## RANSAC
    def run_ransac(self, points_reg, points_warped, fit_function, acceptable_error=2, min_samples=3, fraction_inlier=0.05):
        
        # Rearrange the points in 3 by N arrays of homogeneous representation of the points
        x = np.row_stack((points_reg.transpose(1,0), np.ones((1, points_reg.shape[0]))))
        x_dash = np.row_stack((points_warped.transpose(1, 0), np.ones((1, points_warped.shape[0]))))
        
        #Randomly select the minimum number of corresponding pairs required to determine the model parameters.
        #Estimate the parameters of the model.
        N = x.shape[1]
        best_inliers = np.zeros((N,))
        N_ransac = self.ransac_N(min_samples, fraction_inlier, 0.99)
        
        for i in range(N_ransac):
            indices = np.random.choice(range(N), min_samples, replace=False)
            selected_points = np.zeros((N,))
            selected_points[indices]=1
            T, re = fit_function(x, x_dash, selected_points)
            inliers = self.detect_intliers(x, x_dash, T, acceptable_error) #Determine inliers -- how many points from the set of all points are within an acceptable error/residual.
            
            #Keep track of the sample that results in the maximum number of inliers.
            if(np.sum(inliers) > np.sum(best_inliers)):
                best_inliers = inliers
                T_best = T
        selected_points = np.zeros((N,))
        selected_points[np.nonzero(best_inliers)]=1
        T_best, re = fit_function(x, x_dash, selected_points)
        #T = np.eye(3)
        #re = 0.0
        
        return (T_best, best_inliers, re)


    
    ## Warp a
    def warp_a(self, h, h_inv):
         # Get 4 corner points
        cx, cy = [], []
        for fx in [0, self.width - 1]:
            for fy in [0, self.height - 1]:
                x, y = h(fx, fy)
                x, y = int(x), int(y)
                cx.append(x)
                cy.append(y)

        # Get min and max, then new width and height
        min_x, max_x = int(min(cx)), int(max(cx))
        min_y, max_y = int(min(cy)), int(max(cy))
        width_g = max_x - min_x + 1
        height_g = max_y - min_y + 1

        # Creates empty new image
        img_new = np.zeros((height_g, width_g, self.channels))

        # Find pixel values and map to new image
        for gy in range(min_y, max_y + 1):
            for gx in range(min_x, max_x + 1):
                fx, fy = h_inv(gx, gy)
                #need to have the condition to check h_inv returns a value out of the input plane's range.
                if (fx >self.shape[1] - 1 or fx < 0 or fy < 0 or fy > self.shape[0] - 1):
                    continue
                fx, fy = int(fx), int(fy)
                img_new[gy - min_y, gx - min_x] = self.img[fy, fx]

        # Returns new image
        return img_new 

    ## Warp b
    def warp_b(self, h_inv, output_shape):
         # Create empty new image
        if len(output_shape) < 3:
            output_shape = output_shape + (self.channels,)
        img_new = np.zeros(output_shape)

        # Find pixel values and map to new image
        for gy in range(output_shape[0]):
            for gx in range(output_shape[1]):
                fx, fy = h_inv(gx, gy)
                #need to have the condition to check h_inv returns a value out of the input plane's range.
                if (fx >self.shape[1] - 1 or fx < 0 or fy < 0 or fy > self.shape[0] - 1):
                    continue
                fx, fy = int(fx), int(fy)
                img_new[max(0, gy), max(0, gx)] = self.img[fy, fx]
                

        # Returns new image
        return img_new
    
   
    ## Affine fit function
    def affine_fit(self, x, x_dash, select_flag):
        
       
        # compute the matrix H from point coordinate moments
        M = x @ np.diag(select_flag) @ x.transpose(1, 0)
        H1 = np.column_stack((M, np.zeros((3, 3))))
        H2 = np.column_stack((np.zeros((3, 3)), M))
        H = np.row_stack((H1, H2))
       
        #vector b
        b_dash = x @ np.diag(select_flag) @ (x_dash - x).transpose(1, 0)
        b = np.row_stack((b_dash[:,0][:,None], b_dash[:,1][:,None]))
       # print('b = \n', b)
        
        
        p = self.qr_solve(H, b)
        p = p.squeeze()
       
        # the parameter vector is [a_00, a_01, t_x, a_10, a_11, t_y]
        # rearrange it back into homogeneous matrix representation
        T_affine = np.row_stack((p.reshape(2, 3), [0,0,0])) + np.eye(3)
        
        x_t = T_affine @ x
        res_error = np.sum(np.power((x_dash - x_t)*select_flag, 2))
        return(T_affine, res_error)
        

    ## Rotation and translation fit function
    def rt_fit(self, points_reg, points_warped):

       # create homogeneous representation of input points and turn into torch tensors
        x1 = np.row_stack((points_reg.transpose(1,0), np.ones((1, points_reg.shape[0]))))
        x = torch.tensor(x1, device = device, requires_grad=False)
        
        # torch tensor for the output points
        x_dash = torch.tensor(points_warped.transpose(1,0), device=(device), requires_grad=False)
        
        h_22 = torch.tensor([0], device=(device), requires_grad=False)
        
        #performing affine fit with outlier detection
        T_affine, matches_selected, re_affine = self.run_ransac(points_reg, points_warped, self.affine_fit, acceptable_error=6, min_samples=3, fraction_inlier = 0.2)
        
        matches_selected = torch.tensor(np.diag(matches_selected), dtype=float, device=(device), requires_grad=False)
        
        
        #---------------------the loss function to be optimized-----------------------------
        #Computes the fit error of 2D homography fit between X and X_dash using h_8 parameters
        def fit_error(h_8):
            H = torch.cat((h_8, h_22), 0)
            x_t = torch.matmul(H.reshape(3, 3) + torch.eye(3, device=(device)), x)
            # the identity matrix addition is to keep the parameterization such that
            # h_8 = 0 results in an identity transformation.
            xt_nh = torch.div(x_t, x_t[2,:])
            xt_nh = xt_nh[0:2,:]
            
            xloss = (x_dash - xt_nh) @ matches_selected
            
            return(torch.pow(xloss, 2).sum())
        
        #-----------------------initialize with affine estimate----------------------------------------
        # initialize the 8 parameters of the perspective transform matrix. Recall, h_22 = 0, hence only 8
        h = (T_affine - np.eye(3)).reshape(9,)[0:8]  # we remove an identity matrix from T_affine as per parameterization convention (p=0, represents the identity matrix)
       
        h_8_est = torch.tensor(h, device=(device), requires_grad=True)
        
        
        learning_rate = 1
        prev_res = 99999.0
        exit_flag = False
        t = 0
        
        #------------------------------------estimation iterations----------------------------------------   
        while(exit_flag == False):
            Hessian = torch.autograd.functional.hessian(fit_error, h_8_est)
            
            residual = fit_error(h_8_est)/x.shape[1]  # Residual is a Tensor of shape (1,)
            
            # Use autograd to compute the backward pass. This call will compute the
            # gradient of loss with respect to all Tensors with requires_grad=True
            residual.backward()
            
            # Manually update weights using gradient descent. Wrap in torch.no_grad()
            # because weights have requires_grad=True, but we don't need to track this
            # in autograd.
            
            with torch.no_grad():
                del_p = learning_rate * torch.inverse(Hessian) @ h_8_est.grad
                h_8_est -= del_p
                if prev_res > residual.item():
                    prev_res = residual.item()
                else:
                    exit_flag = True
                
                h_8_est.grad = None
            t = t+1
            if(t>2000):
                exit_flag = True
        
        T_per = torch.cat((h_8_est, h_22), 0).reshape(3, 3) + torch.eye(3, device=device)
        Trans_per = T_per.detach().cpu().numpy()
        re_per = torch.sqrt(residual)

        # we have projective, now we have to calculate the affine by using 
        #
        #                          [[1,   0,   0]
        # T_projectve = T_affine *  [0,   1,   0]
        #                           [h_7, h_8, 1]
      
        h7 = Trans_per[2][0]  # a[2][0] coefficient of proejctive 
        h8 = Trans_per[2][1]  # a[2][1] coefficient of proejctive 
        

        temp_mat = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [h7, h8, 1]])
        # since we need to calculate the affine, we have to take inverse then matrix multiplication
        inv_temp_mat = np.linalg.inv(temp_mat)
        
        aff_trans = np.matmul(Trans_per, inv_temp_mat) # matrix multiplication of projectiona adn temp_mat

        # getting theta by normalizing the first column of the affine transform, i,e. divide by sqrt(a_00^2 + a_10^2) and then equate to cos(theta)

        #cos (theta) = a_00/ sqrt(a_00^2 + a_10^2) 
        #theta = invcos (a_00/ sqrt(a_00^2 + a_10^2) )

        theta = math.acos(aff_trans[0][0] / math.sqrt((aff_trans[0][0]*aff_trans[0][0]) + (aff_trans[1][0]*aff_trans[1][0])))

        #final RT matrix
        T_rt = np.array([[np.cos(theta), -np.sin(theta), aff_trans[0][2]], 
                          [np.sin(theta), np.cos(theta), aff_trans[1][2]], 
                          [0, 0, 1]])

        return(T_rt, re_per.detach().numpy())


    ## Perspective/Projection fit function
    #Newton's method
    
    def perspective_fit(self, points_reg, points_warped): 
        
       # create homogeneous representation of input points and turn into torch tensors
        x1 = np.row_stack((points_reg.transpose(1,0), np.ones((1, points_reg.shape[0]))))
        x = torch.tensor(x1, device = device, requires_grad=False)
        
        # torch tensor for the output points
        x_dash = torch.tensor(points_warped.transpose(1,0), device=(device), requires_grad=False)
        
        h_22 = torch.tensor([0], device=(device), requires_grad=False)
        
        #performing affine fit with outlier detection
        T_affine, matches_selected, re_affine = self.run_ransac(points_reg, points_warped, self.affine_fit, acceptable_error=6, min_samples=3, fraction_inlier = 0.2)
        
        matches_selected = torch.tensor(np.diag(matches_selected), dtype=float, device=(device), requires_grad=False)
        
        
        #---------------------the loss function to be optimized-----------------------------
        #Computes the fit error of 2D homography fit between X and X_dash using h_8 parameters
        def fit_error(h_8):
            H = torch.cat((h_8, h_22), 0)
            x_t = torch.matmul(H.reshape(3, 3) + torch.eye(3, device=(device)), x)
            # the identity matrix addition is to keep the parameterization such that
            # h_8 = 0 results in an identity transformation.
            xt_nh = torch.div(x_t, x_t[2,:])
            xt_nh = xt_nh[0:2,:]
            
            xloss = (x_dash - xt_nh) @ matches_selected
            
            return(torch.pow(xloss, 2).sum())
        
        #-----------------------initialize with affine estimate----------------------------------------
        # initialize the 8 parameters of the perspective transform matrix. Recall, h_22 = 0, hence only 8
        h = (T_affine - np.eye(3)).reshape(9,)[0:8]  # we remove an identity matrix from T_affine as per parameterization convention (p=0, represents the identity matrix)
       
        h_8_est = torch.tensor(h, device=(device), requires_grad=True)
        
        
        learning_rate = 1
        prev_res = 99999.0
        exit_flag = False
        t = 0
        
        #------------------------------------estimation iterations----------------------------------------   
        while(exit_flag == False):
            Hessian = torch.autograd.functional.hessian(fit_error, h_8_est)
            
            residual = fit_error(h_8_est)/x.shape[1]  # Residual is a Tensor of shape (1,)
            
            # Use autograd to compute the backward pass. This call will compute the
            # gradient of loss with respect to all Tensors with requires_grad=True
            residual.backward()
            
            # Manually update weights using gradient descent. Wrap in torch.no_grad()
            # because weights have requires_grad=True, but we don't need to track this
            # in autograd.
            
            with torch.no_grad():
                del_p = learning_rate * torch.inverse(Hessian) @ h_8_est.grad
                h_8_est -= del_p
                if prev_res > residual.item():
                    prev_res = residual.item()
                else:
                    exit_flag = True
                
                h_8_est.grad = None
            t = t+1
            if(t>2000):
                exit_flag = True
        
        T_per = torch.cat((h_8_est, h_22), 0).reshape(3, 3) + torch.eye(3, device=device)  # perspective matrix in torch.tensor form
        Trans_per = T_per.detach().cpu().numpy()  # perspective in matrix form
        re_per = torch.sqrt(residual)   #res. error in torch.tensure form

        return(Trans_per, re_per.detach().numpy())
    



    ##### STUDENTS SECTION #####
    ## Do not change the functions definition or return types

    ## Affine fit
    def fit_affine(self):
        # Can delete, just makes blank residual and Transform matrix
        points_reg, points_warped = self.get_sift_points(num_matches=8)
        T, matches_selected, re = self.run_ransac(points_reg, points_warped, self.affine_fit, acceptable_error=6, min_samples=3, fraction_inlier = 0.2)
      
        # Dont change output/return
        return (T, re)

    ##  Rotation & Translation fit
    def fit_rotate_translation(self):
        # Can delete, just makes blank residual and Transform matrix
        points_reg, points_warped = self.get_sift_points(num_matches=8)
        T, re = self.rt_fit(points_reg, points_warped)
        
        # Dont change output/return
        return (T, re)

    ## Perspective/Projective fit
    def fit_perspective(self):
        # Can delete, just makes blank residual and Transform matrix
        points_reg, points_warped = self.get_sift_points(num_matches=40)
        T, re = self.perspective_fit(points_reg, points_warped)
        # Dont change output/return
        return (T, re)