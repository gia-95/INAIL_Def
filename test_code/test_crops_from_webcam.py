import cv2
import utils_webcam
import utils


### COFIGURAZIONI
HEIGHT_WINDOWSHOW = 600
WIDTH_CORPO = 500
WIDTH_TESTA = 300
WIDTH_MANO = 200
WIDTH_TUTA = 300
WIDTH_STIVALE = 200
VIDEO_DIR = 'video_webcam'
VIDEO_NAME = 'cascoNO'

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object (e dove salvo il video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


while True:
    ret, frame = cam.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    # Frame variable
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    ### Detect Mediapipe
    annotated_image, detection_result = utils_webcam.get_lanmarked_image(frame)
  
    # SHOW MEDIAPIPE
    annotated_image = cv2.resize(annotated_image, (WIDTH_CORPO, round((WIDTH_CORPO*annotated_image.shape[0])/annotated_image.shape[1])))
    cv2.namedWindow("mediapipe", cv2.WINDOW_NORMAL)
    cv2.imshow("mediapipe", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.resizeWindow("mediapipe", WIDTH_CORPO, round((WIDTH_CORPO*annotated_image.shape[0])/annotated_image.shape[1]))

    ### DISPLAY CROPS
    if (len(detection_result.pose_landmarks) != 0) :  
        
        # TESTA
        crop_testa = utils.crop_testa_from_image(detection_result, frame)
        if (crop_testa is not None) :
            crop_testa = cv2.resize(crop_testa, (WIDTH_TESTA, round((WIDTH_TESTA*crop_testa.shape[0])/crop_testa.shape[1])))
            cv2.namedWindow("testa", cv2.WINDOW_NORMAL)
            cv2.imshow("testa", crop_testa)
            cv2.resizeWindow("testa", WIDTH_TESTA, round((WIDTH_TESTA*crop_testa.shape[0])/crop_testa.shape[1]))
            cv2.moveWindow("testa", WIDTH_CORPO, -100)
        else :
            cv2.destroyWindow("testa")
            
        # MANO DX
        crop_mano_dx = utils.crop_mano_dx_from_image(detection_result, frame)
        if (crop_mano_dx is not None) :
            crop_mano_dx = cv2.resize(crop_mano_dx, (WIDTH_MANO, round((WIDTH_MANO*crop_mano_dx.shape[0])/crop_mano_dx.shape[1])))
            cv2.namedWindow("mano_dx", cv2.WINDOW_NORMAL)
            cv2.imshow("mano_dx", crop_mano_dx)
            cv2.resizeWindow("mano_dx", WIDTH_MANO, round((WIDTH_MANO*crop_mano_dx.shape[0])/crop_mano_dx.shape[1]))
            cv2.moveWindow("mano_dx", WIDTH_CORPO + WIDTH_TESTA, -100)
        else :
            cv2.destroyWindow("mano_dx")
            
        # MANO SX
        crop_mano_sx = utils.crop_mano_sx_from_image(detection_result, frame)
        if (crop_mano_sx is not None) :
            crop_mano_sx = cv2.resize(crop_mano_sx, (WIDTH_MANO, round((WIDTH_MANO*crop_mano_sx.shape[0])/crop_mano_sx.shape[1])))
            cv2.namedWindow("mano_sx", cv2.WINDOW_NORMAL)
            cv2.imshow("mano_sx", crop_mano_sx)
            cv2.resizeWindow("mano_sx", WIDTH_MANO, round((WIDTH_MANO*crop_mano_sx.shape[0])/crop_mano_sx.shape[1]))
            cv2.moveWindow("mano_sx", WIDTH_CORPO + WIDTH_TESTA + WIDTH_MANO + 10, -100)
        else :
            cv2.destroyWindow("mano_sx")
            
        # TUTA
        crop_tuta = utils.crop_tuta_from_image(detection_result, frame)
        if ((crop_tuta is not None) and (crop_tuta is not (None, None))) :
            crop_tuta = cv2.resize(crop_tuta, (WIDTH_TUTA, round((WIDTH_TUTA*crop_tuta.shape[0])/crop_tuta.shape[1])))
            cv2.namedWindow("tuta", cv2.WINDOW_NORMAL)
            cv2.imshow("tuta", crop_tuta)
            cv2.resizeWindow("tuta", WIDTH_TUTA, round((WIDTH_TUTA*crop_tuta.shape[0])/crop_tuta.shape[1]))
            cv2.moveWindow("tuta", WIDTH_CORPO, 300)
        else :
            cv2.destroyWindow("tuta")
            
        # STIVALE DX
        crop_stivale_dx = utils.crop_stivali_dx_from_image(detection_result, frame)
        if ((crop_stivale_dx is not None) and (crop_stivale_dx is not (None, None))) :
            crop_stivale_dx = cv2.resize(crop_stivale_dx, (WIDTH_STIVALE, round((WIDTH_STIVALE*crop_stivale_dx.shape[0])/crop_stivale_dx.shape[1])))
            cv2.namedWindow("stivale_dx", cv2.WINDOW_NORMAL)
            cv2.imshow("stivale_dx", crop_stivale_dx)
            cv2.resizeWindow("stivale_dx", WIDTH_STIVALE, round((WIDTH_STIVALE*crop_stivale_dx.shape[0])/crop_stivale_dx.shape[1]))
            cv2.moveWindow("stivale_dx", WIDTH_CORPO + WIDTH_TESTA, 300)
        else :
            cv2.destroyWindow("stivale_dx")
            
        # STIVALE SX
        crop_stivale_sx = utils.crop_stivali_sx_from_image(detection_result, frame)
        if ((crop_stivale_sx is not None) and (crop_stivale_sx is not (None, None))) :
            crop_stivale_sx = cv2.resize(crop_stivale_sx, (WIDTH_STIVALE, round((WIDTH_STIVALE*crop_stivale_sx.shape[0])/crop_stivale_sx.shape[1])))
            cv2.namedWindow("stivale_sx", cv2.WINDOW_NORMAL)
            cv2.imshow("stivale_sx", crop_stivale_sx)
            cv2.resizeWindow("stivale_sx", WIDTH_STIVALE, round((WIDTH_STIVALE*crop_stivale_sx.shape[0])/crop_stivale_sx.shape[1]))
            cv2.moveWindow("stivale_sx",  WIDTH_CORPO + WIDTH_TESTA + WIDTH_STIVALE + 20, 300)
        else :
            cv2.destroyWindow("stivale_sx")
               
    
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
    
    