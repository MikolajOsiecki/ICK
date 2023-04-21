import numpy as np
import cv2 as cv
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
    def __init__(self, template_path = 'matching/', img_path = (str), min_match_count = 70):
        self.template_path = template_path
        self.img_path = img_path
        self.min_match_count = min_match_count
        self.sift = cv.SIFT_create() # Initiate SIFT detector

    def match_phots(self):
        """Takes in two images and compares them using SIFT and FLANN,
        outputs the number of matches and the score of the match.

        Returns:
            score: number of matches divided by the number of templates
            match: number of matches
        """
        # img1 = cv.imread('matching/kubek.jpg', cv.IMREAD_GRAYSCALE) #static path for single queryImage, not used
        img2 = cv.imread(self.img_path, cv.IMREAD_GRAYSCALE) # trainImage
        templates = glob.glob(str(self.template_path)+'*.jpg')
        self.score = 0 # reset score
        self.match = 0 # reset match
        for template in templates:
            img = cv.imread(template, cv.IMREAD_GRAYSCALE)

            # find the keypoints and descriptors with SIFT
            kp1, des1 = self.sift.detectAndCompute(img,None)
            kp2, des2 = self.sift.detectAndCompute(img2,None)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1,des2,k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)

            # process only if enough matches are found
            if len(good)>self.min_match_count:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
                # matchesMask = mask.ravel().tolist() # needed only for comparison
                h,w = img.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv.perspectiveTransform(pts,M)
                img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

                # plt.imshow(img2, 'gray'),plt.show() # show image
                x1 = dst[0][0][0]
                y1 = dst[0][0][1]              
                #print coordinates of the left top corner
                print("x1: " + str(x1))
                print("y1: " + str(y1))

                self.score += (len(good))/len(templates)
                self.match += 1
                print( f"Match found -" + template )
            else:
                print( "Not enough matches are found - {}/{}".format(len(good), self.min_match_count) )
                # matchesMask = None # needed only for comparison
                self.score += (len(good))/len(templates)
                self.match += 0

            # display comparison with matched features
            # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
            #                 singlePointColor = None,
            #                 matchesMask = matchesMask, # draw only inliers
            #                 flags = 2)
            # img3 = cv.drawMatches(img,kp1,img2,kp2,good,None,**draw_params)
            # plt.imshow(img3, 'gray'),plt.show()
        return self.score, self.match


