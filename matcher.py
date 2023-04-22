import numpy as np
import cv2 
import glob
# from matplotlib import pyplot as plt #only needed for comparison

class Matcher:
    """
    Class for matching photos using SIFT and FLANN
    
    Args:
        template_path: path to templates
        img_path: path to image to be matched
        min_match_count: minimum number of matches to be considered a match
    """
    def __init__(self, name = "Matcher", template_path = 'matching/', img_path = (str), min_match_count = 70):
        self.name = name
        self.template_path = template_path
        self.img_path = img_path
        self.min_match_count = min_match_count
        self.sift = cv2.SIFT_create() # Initiate SIFT detector
        self.x, self.y = 0, 0 # coordinates

    def match_phots(self):
        """Takes in two images and compares them using SIFT and FLANN,
        outputs the number of matches and the score of the match.

        Returns:
            score: number of matches divided by the number of templates
            match: number of matches
        """
        # img1 = cv.imread('matching/kubek.jpg', cv.IMREAD_GRAYSCALE) #static path for single queryImage, not used
        img2 = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE) # trainImage
        templates = glob.glob(str(self.template_path)+'*.jpg')
        self.score = 0 # reset score
        self.match = 0 # reset match
        for template in templates:
            img = cv2.imread(template, cv2.IMREAD_GRAYSCALE)

            # find the keypoints and descriptors with SIFT
            kp1, des1 = self.sift.detectAndCompute(img,None)
            kp2, des2 = self.sift.detectAndCompute(img2,None)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1,des2,k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)

            # process only if enough matches are found
            if len(good)>self.min_match_count:
                # print(template + " - Good matches: " + str(len(good)))
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist() # needed only for comparison
                h,w = img.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

                x1 = dst[0][0][0]
                y1 = dst[0][0][1]         
                x2 = dst[1][0][0]
                y2 = dst[1][0][1]   
                x3 = dst[2][0][0]   
                y3 = dst[2][0][1]
                x4 = dst[3][0][0]
                y4 = dst[3][0][1]
                     
                #get coordinates of center   
                self.x = np.floor((x1 + x2 + x3 + x4)/4)
                self.y = np.floor((y1 + y2 + y3 + y4)/4)
                # print("x: " + str(self.x))

                self.score += (len(good))/len(templates)
                self.match += 1
                # print( f"Match found -" + template )
            else:
                # print( "Not enough matches are found - {}/{}".format(len(good), self.min_match_count) )
                matchesMask = None # needed only for comparison
                self.score += (len(good))/len(templates)
                self.match += 0

            # display comparison with matched features
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = 2)
            img3 = cv2.drawMatches(img,kp1,img2,kp2,good,None,**draw_params)
            cv2.imwrite(self.img_path + self.name + ".jpg", img3)
        return self.score, self.match

    def get_coordinates(self):
        """Returns coordinates of the center of the match.

        Returns:
            x: x coordinate of the center of the match
            y: y coordinate of the center of the match
        """
        return (self.x, self.y)


