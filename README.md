# ICK
ISS ICK

To start appliaction simply run main.py.

To take calibration photos before obstacle detection set take_pictures to False.

When take_pictures is True during calibration, then application will try to take pictures using camera port marked by cam_port. Photo count is equal to picture_count and its save path is set by calib_path.

Afterwards, threads for obstacle picture taking and analysis will be created and started.
Photo thread takes obstacle_path as location in which picture will be saved as well as parameters taken from camera calibration procedure.
Matching threads need path to template location (separate for each thread) which contains only photos to which new photo will be compared to match features (recommended to not exceed more than 3 photos, as it takes a while to compare each photo), direct path to photo taken by PhotoThread also needed. min_match_count is minimum amount of matches found by feature matching alogorythm to count as a match.

Type "q" and press enter in command line to stop the app, this will set stop event for Photo thread and Matching thread as well as set PhotoEvent event to avoid getting stuck in wait for new photo.
