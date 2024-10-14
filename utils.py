# General library
import cv2
import numpy as np
import os

# Mediapipe
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# import matplotlib.pyplot as plt


""""""""" CONFIGURAZIONI """""""""""

#### Instazia il Detector una volta sola ###
base_options = python.BaseOptions(model_asset_path="modelli/mediapipe/pose_landmarker_heavy.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options, 
    output_segmentation_masks=True
)
detector = vision.PoseLandmarker.create_from_options(options) 

""""""" SHOW A SCHERMO IMMAGINI CROPS"""

def printa_immagini(detection_result, image, cropped_stivale_sx, cropped_stivale_dx, TIME_SLEEP = 200) :
    ### Visualizza intero frame diseganto
    cv2.imshow("frame", draw_landmarks_on_image(image, detection_result))
    
    if (cropped_stivale_sx is not None) :
            cv2.imshow("stivale_SX", cropped_stivale_sx)
            cv2.moveWindow("stivale_SX", 300, 500)
            
    if (cropped_stivale_dx is not None) :
        cv2.imshow("stivale_DX", cropped_stivale_dx)
        cv2.moveWindow("stivale_DX", 0, 500)
        
    key = cv2.waitKey(TIME_SLEEP)#pauses for tot seconds before fetching next image
    if key == 27:#if ESC is pressed, exit loop
        cv2.destroyAllWindows()
        return True


""" RIMUOVE IMMAGINI SIMILI (dato Treshold) DALLA CARTELLA PASSATA """

def remove_similar_images_from_directory(
    image_directory, treshold_similarity=50, img_shape=(199, 216)
):
    directory = image_directory
    TRESH_DIFF_IMGS = treshold_similarity

    ### CARICA TUTTE LE IMMAGINI (salvale nell'array 'images[]')
    images = []
    for img_name in os.listdir(directory):
        if img_name == ".DS_Store":
            continue
        img_path = os.path.join(directory, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        # if img.shape != img_shape : print(img.shape,  img_path)
        images.append((img_name, img))
    print("Totale immagini nella cartella:", len(images))

    ### TROVA LE IMAGINI SIMILI SALVANDO GLI INDICI E I NOMI NEI RISPETTIVI ARRAY (seguenti)
    list_idx_to_remove = []
    list_img_name_to_remove = []

    for current_idx in range(len(images) - 1):
        # Se è l'ultima immagine non la controllo con niente
        if current_idx == (len(images) - 1):
            break

        # Se l'immagine è già presente tra quelle da eliminare vado avanti
        if current_idx in list_idx_to_remove:
            continue

        for i in range(current_idx + 1, len(images) - 1):
            # Se il nome del video è diverso vado avanti (confronto solo quelli con stesso video)
            if images[current_idx][0][:-14] != images[i][0][:-14]:
                continue
            # if (images[current_idx][0].split('manodx')[0] != images[i][0].split('manodx')[0]) : continue

            if i in list_idx_to_remove:
                continue

            difference, _ = images_difference_abs(images[current_idx][1], images[i][1])

            if difference < TRESH_DIFF_IMGS:
                list_idx_to_remove.append(i)
                list_img_name_to_remove.append(images[i][0])

    print("Immagini da eliminare:", len(list_idx_to_remove))
    print("Immagini rimanenti (buone):", len(images) - len(list_idx_to_remove))

    ### RIMUOVI DALLA CARTELLA IMMAGINI SIMILI TROVATE
    for img_name in list_img_name_to_remove:
        img_path = os.path.join(directory, img_name)
        # print("- rimuovo:", img_name)
        os.remove(img_path)

def images_difference_abs(img1, img2):
    h = img1.shape[0]
    w = img1.shape[1]
    # diff = cv2.subtract(img1, img2) # se img 1 canale (b&w)
    diff = abs(img1 - img2)
    err = np.sum(diff**2)
    mse = err / (float(h * w))
    return mse, diff


""" ITERAZIONI SU VIDEO-FRAME """
def get_lanmarked_image (image) :
    # STEP 3: Load the input image.
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(mp_image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

    return annotated_image, detection_result

def get_poselandmarker_from_image(image, mediapipe_task="heavy"):    

    # STEP 3: Load the input image.
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(mp_image)
    # print(detection_result)

    # STEP 5: Process the detection result. In this case, visualize it.
    # annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    # cv2.imshow("c", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey()

    return detection_result

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image

def saveImage(image, file_name, dir_path):
    file_path = os.path.join(dir_path, f"{file_name}.jpg")
    cv2.imwrite(file_path, image)
    print(f"-> Salvataggio '{file_path}' effettuato!")

#### Crop TUTA  ####

def crop_tuta_from_image(detection_result, image, return_coord=False):
    # Prendo la lista dei punti del corpo rilevati ([0] perchè è la prima/unica persona)
    pose_landmarks_list = detection_result.pose_landmarks
    pose_landmarks = pose_landmarks_list[0]

    # Height and width image
    height = image.shape[0]
    width = image.shape[1]
    
    ################################### Logica crop ###################################
    ### x : spalle                                                                    #
    ### y : alto -> 50 px sopra le spalle                                             #
    ###     basso -> 200 px sotto l'anca (più o meno sotto il cavallo)                #
    ###################################################################################

    if not checkCorrectPositionTuta(pose_landmarks, height, width):
        return None, None
    
    try :
        coord_spallaSx = pose_landmarks[11]
        coord_spallaDx = pose_landmarks[12]
        coord_ancaSx = pose_landmarks[23]
        coord_ancaDx = pose_landmarks[24]
        
        x_min = max(0, coord_spallaDx.x * width)
        x_max = min(width, coord_spallaSx.x * width)
        y_min = max(0, min((coord_spallaSx.y * height, coord_spallaDx.y * height)) - 50)
        y_max = min(height, max(coord_ancaSx.y * height, coord_ancaDx.y * height) + 200)

        if  x_min <= x_max: # Solo come controllo che non sto di spalle
            crop = image[round(y_min):round(y_max), round(x_min):round(x_max)]
            return crop, (round(y_min), round(y_max), round(x_min), round(x_max))
        else:
            return None, None
    except:  
        return None, None

def checkCorrectPositionTuta(pose_landmarks, w_img, h_img):
    coord_manoSx = pose_landmarks[17]
    coord_manoDx = pose_landmarks[18]
    coord_spallaSx = pose_landmarks[11]
    coord_spallaDx = pose_landmarks[12]
    coord_ancaSx = pose_landmarks[23]
    coord_ancaDx = pose_landmarks[24]

    if (
        # Le mani devono stare all'esterno delle anche
        ((coord_manoDx.x * w_img > coord_ancaDx.x * w_img) and (coord_manoDx.y*h_img > coord_spallaDx.y*h_img) and (coord_manoDx.y*h_img < coord_ancaDx.y*h_img))
        or
        ((coord_manoSx.x * w_img < coord_ancaSx.x * w_img) and (coord_manoSx.y*h_img > coord_spallaSx.y*h_img) and (coord_manoSx.y*h_img < coord_ancaSx.y*h_img))
        or
        # Non sto mezzo piegato a raccogliere qualcosa (le 'y' delle spalle non sono simili)
        abs(coord_spallaDx.y * h_img - coord_spallaSx.y * h_img) >= 70
        or
        # Non stò mezzo girato da una parte (le 'z' delle mani devono essere simili)
        abs(coord_spallaDx.z - coord_spallaSx.z) > 0.5
    ):
        return False
    else:
        return True

#### Crop TESTA ####

def crop_testa_from_image(detection_result, image):
    
    ################################### Logica crop ###################################
    ### x : Cambia a seconda se il soggetto sta di faccia, girato a dx o girato a sx  #
    ### y : alto -> naso + 150px                                                      #
    ###     basso -> spalle + 100 px                                                  #
    ###################################################################################
    
    # Prendo la lista dei punti del corpo rilevati ([0] perchè è la prima/unica persona)
    pose_landmarks_list = detection_result.pose_landmarks
    pose_landmarks = pose_landmarks_list[0]
    
    # Height and width image
    height = image.shape[0]
    width = image.shape[1]
    
    if not checkCorrectPositionTesta(pose_landmarks, height, width):
        return None, None

    try :
        # Coordinate punti del corpo
        coord_spallaSx = pose_landmarks[11]
        coord_spallaDx = pose_landmarks[12]
        coord_naso = pose_landmarks[0]

        # Girato verso destra
        if (coord_spallaDx.z - coord_spallaSx.z > 0.5) :
            x_min = max(0, coord_naso.x*width - 100)
            x_max = min(width, coord_naso.x*width + 200)
        
        # Girato verso sinistra
        elif (coord_spallaDx.z - coord_spallaSx.z < -0.5) :
            x_min = max(0, coord_naso.x*width - 200)
            x_max = min(width, coord_naso.x*width + 100)
        
        # Di faccia
        else :
            x_min = max(0, coord_naso.x*width - 150)
            x_max = min(width, coord_naso.x*width + 150)
        
        
        y_min = max(0, coord_naso.y*height - 250)
        y_max = min(height, max(coord_spallaSx.y*height, coord_spallaDx.y*height))
        
        crop = image[round(y_min) : round(y_max), round(x_min) : round(x_max)]
    except :
        return None, None
    
    return crop , (round(y_min) , round(y_max), round(x_min), round(x_max))
        
def checkCorrectPositionTesta(pose_landmarks, w_img, h_img):
    coord_spallaSx = pose_landmarks[11]
    coord_spallaDx = pose_landmarks[12]
    coord_manoSx = pose_landmarks[17]
    coord_manoDx = pose_landmarks[18]
    coord_boccaSx = pose_landmarks[10]
    coord_boccaDx = pose_landmarks[9]
    coord_occhioSx = pose_landmarks[2]
    coord_occhioDx = pose_landmarks[5]
    coord_naso = pose_landmarks[0]
    
    # Presence
    if (coord_boccaSx.presence < 0.9 or
        coord_boccaDx.presence < 0.9 or
        coord_occhioSx.presence < 0.9 or
        coord_occhioDx.presence < 0.9 or
        coord_naso.presence < 0.9
        ):
            return False
    
    # Se la bocca sta sotto le spalle (strano) -> non corretto
    if ((coord_boccaDx.y*h_img > coord_spallaSx.y*h_img) or (coord_boccaDx.y*h_img > coord_spallaDx.y*h_img) 
        or (coord_boccaSx.y*h_img > coord_spallaSx.y*h_img) or (coord_boccaSx.y*h_img > coord_spallaDx.y*h_img)):
        return False
         
    if (
        # Mano DX all'interno della faccia                       ...e più alta delle spalle
        (coord_manoDx.x * w_img >= coord_spallaDx.x * w_img) and (coord_manoDx.y * h_img <= coord_spallaDx.y * h_img)
        or
        # oppure mano SX all'interno della faccia                ...e più alta delle spalle
        (coord_manoSx.x * w_img <= coord_spallaSx.x * w_img) and (coord_manoSx.y * h_img <= coord_spallaSx.y * h_img)):
            return False
    else:
            return True

### Crop MANO DX ###

def crop_mano_dx_from_image(detection_result, image, return_coord=False):
    # Prendo la lista dei punti del corpo rilevati ([0] perchè è la prima/unica persona)
    pose_landmarks_list = detection_result.pose_landmarks
    pose_landmarks = pose_landmarks_list[0]

    # Height and width image
    height = image.shape[0]
    width = image.shape[1]
    
    if not checkCorrectPositionManoDx(pose_landmarks, height, width):
        return None, None
    
    try :
        
        # Coordinate punti del corpo
        coord_manoDx_bassa = pose_landmarks[18]
        coord_manoDx_alta = pose_landmarks[20]
        coord_gomitoDx = pose_landmarks[14]
        
        ### In base alla posizione delle mani faccio il crop (in giù, in sù, ecc.)
        
        # Mano in giù (sotto il gomito)
        if ((coord_manoDx_bassa.y*height - coord_gomitoDx.y*height) > 100) :
            x_min = max(0, coord_manoDx_bassa.x*width - 150)
            x_max = min(width, coord_manoDx_bassa.x*width + 150)
            y_min = max(0, coord_gomitoDx.y*height)
            y_max = min(height, coord_manoDx_bassa.y*height + 150)
            
        # Mano in sù (sopra il gomito)
        elif ((coord_manoDx_bassa.y*height - coord_gomitoDx.y*height) < -100) :
            x_min = max(0, coord_manoDx_bassa.x*width - 150)
            x_max = min(width, coord_manoDx_bassa.x*width + 150)
            y_min = max(0, coord_manoDx_bassa.y*height - 150)
            y_max = min(height, coord_gomitoDx.y*height)
            
        # Mano altezza del gomito
        else : 
            # braccio interno al corpo
            if (coord_manoDx_bassa.x*width > coord_gomitoDx.x*width) :
                x_min = max(0, coord_gomitoDx.x*width)
                x_max = min(width, coord_manoDx_bassa.x*width + 150)
                y_min = max(0, coord_manoDx_alta.y*height - 150)
                y_max = min(height, coord_manoDx_bassa.y*height + 120)
                
            # Braccio esterno al corpo
            else :
                x_min = max(0, coord_manoDx_bassa.x*width - 150)
                x_max = min(width, coord_gomitoDx.x*width)
                y_min = max(0, coord_manoDx_alta.y*height - 100)
                y_max = min(height, coord_manoDx_bassa.y*height + 120)
                
        crop_manoDx = image[round(y_min) : round(y_max), round(x_min) :round(x_max)]
        
        return crop_manoDx, (round(y_min) , round(y_max), round(x_min) , round(x_max))
    
    except :
        
        return None, None

def checkCorrectPositionManoDx(pose_landmarks, height, width) :
    coord_manoSx = pose_landmarks[17]
    coord_manoDx = pose_landmarks[18]
    coord_spallaSx = pose_landmarks[11]
    coord_spallaDx = pose_landmarks[12]
    coord_polsoDx = pose_landmarks[16]
    coord_gomitoDx = pose_landmarks[14]
    
    # Presence
    if (coord_manoDx.presence < 0.9 or
        coord_polsoDx.presence < 0.9 ):
            return False
    
    # Mano protesa in avanti NO
    if ((abs(coord_gomitoDx.x*width - coord_polsoDx.x*width) < 50) and (abs(coord_gomitoDx.y*height - coord_polsoDx.y*height) < 50)) :
        return False
    
    # Girato verso sinistra (la mano destra non si vede) NO
    if ((coord_spallaSx.z - coord_spallaDx.z) < -0.5) :
        return False

    # Se di fronte (primi due if), mani sovrapposte (secondi due if) NO
    if (
        abs(coord_spallaDx.z - coord_spallaSx.z) > -0.5 and
        abs(coord_spallaDx.z - coord_spallaSx.z) < 0.5 and
        (abs(coord_manoDx.x*width - coord_manoSx.x*width) < 200) and
        (abs(coord_manoDx.y*height - coord_manoSx.y*height) < 150)) :
            return False

    return True

### Crop MANO SX ###

def crop_mano_sx_from_image(detection_result, image, return_coord=False):
    # Prendo la lista dei punti del corpo rilevati ([0] perchè è la prima/unica persona)
    pose_landmarks_list = detection_result.pose_landmarks
    pose_landmarks = pose_landmarks_list[0]

    # Height and width image
    height = image.shape[0]
    width = image.shape[1]
    
    if not checkCorrectPositionManoSx(pose_landmarks, height, width):
        return None, None
    
    # Coordinate punti del corpo
    coord_manoSx_bassa = pose_landmarks[17]
    coord_manoSx_alta = pose_landmarks[19]
    coord_gomitoSx = pose_landmarks[13]
    
    try :

        # Mano in giù (sotto il gomito)
        if ((coord_manoSx_bassa.y*height - coord_gomitoSx.y*height) > 100) :
            x_min = max(0, coord_manoSx_bassa.x*width - 150)
            x_max = min(width, coord_manoSx_bassa.x*width + 150)
            y_min = max(0, coord_gomitoSx.y*height)
            y_max = min(height, coord_manoSx_bassa.y*height + 150)
            
        # Mano in sù (sopra il gomito)
        elif ((coord_manoSx_bassa.y*height - coord_gomitoSx.y*height) < -100) :
            x_min = max(0, coord_manoSx_bassa.x*width - 150)
            x_max = min(width, coord_manoSx_bassa.x*width + 150)
            y_min = max(0, coord_manoSx_bassa.y*height - 150)
            y_max = min(height, coord_gomitoSx.y*height)
            
        # Mano altezza del gomito
        else : 
            # braccio interno al corpo
            if (coord_manoSx_bassa.x*width < coord_gomitoSx.x*width) :
                x_min = max(0, coord_manoSx_bassa.x*width - 150)
                x_max = min(width, coord_gomitoSx.x*width)
                y_min = max(0, coord_manoSx_alta.y*height - 150)
                y_max = min(height, coord_manoSx_bassa.y*height + 120)
                
            # Braccio esterno al corpo
            else :
                x_min = max(0, coord_gomitoSx.x*width)
                x_max = min(width, coord_manoSx_bassa.x*width + 150)
                y_min = max(0, coord_manoSx_alta.y*height - 100)
                y_max = min(height, coord_manoSx_bassa.y*height + 120)
                
        crop_manoDx = image[round(y_min) : round(y_max), round(x_min) :round(x_max)]
        
        return crop_manoDx, (round(y_min) , round(y_max), round(x_min) , round(x_max))
    
    except :
        
        return None, None    

def checkCorrectPositionManoSx(pose_landmarks, height, width) :
    coord_manoSx = pose_landmarks[17]
    coord_manoDx = pose_landmarks[18]
    coord_spallaSx = pose_landmarks[11]
    coord_spallaDx = pose_landmarks[12]
    coord_polsoSx = pose_landmarks[15]
    coord_gomitoSx = pose_landmarks[13]
    
    # Presence
    if (coord_manoSx.presence < 0.9 or
        coord_polsoSx.presence < 0.9 ):
            return False
    
    # Mano protesa in avanti NO
    if ((abs(coord_gomitoSx.x*width - coord_polsoSx.x*width) < 50) and (abs(coord_gomitoSx.y*height - coord_polsoSx.y*height) < 50)) :
        return False
    
     # Girato verso sinistra (la mano destra non si vede) NO
    if ((coord_spallaSx.z - coord_spallaDx.z) > 0.5) :
        return False
    
    # Se di fronte (primi due if), mani sovrapposte (secondi due if) NO
    if (
        abs(coord_spallaDx.z - coord_spallaSx.z) > -0.5 and
        abs(coord_spallaDx.z - coord_spallaSx.z) < 0.5 and
        (abs(coord_manoDx.x*width - coord_manoSx.x*width) < 200) and
        (abs(coord_manoDx.y*height - coord_manoSx.y*height) < 150)) :
            return False

    return True

### Crop STIVALI ###

def crop_stivali_sx_from_image(detection_result, image, return_coord=False) :

    # Prendo la lista dei punti del corpo rilevati ([0] perchè è la prima/unica persona)
    pose_landmarks_list = detection_result.pose_landmarks
    pose_landmarks = pose_landmarks_list[0]
    
    # Height and width image
    height = image.shape[0]
    width = image.shape[1]
    
    if (not checkCorrectPositionStivaleSx(pose_landmarks, height, width)):
        return None, None
    
    try:
        
        # Coordinate punti dei piedi SX
        caviglia_piede_SX = pose_landmarks[27]
        tallone_piede_SX = pose_landmarks[29]
        punta_piede_SX = pose_landmarks[31]
        
        x_min_SX = max(0, min(tallone_piede_SX.x*width, punta_piede_SX.x*width) - 60)
        x_max_SX = min(width, max(tallone_piede_SX.x*width, punta_piede_SX.x*width) + 60)
        y_min_SX = max(0, caviglia_piede_SX.y*height - 100)
        y_max_SX = min(height, max(punta_piede_SX.y*height, tallone_piede_SX.y*height) + 100)
        
        crop_stivale_SX = image[round(y_min_SX) : round(y_max_SX) , round(x_min_SX) : round(x_max_SX)]
        
        return crop_stivale_SX, (round(y_min_SX) , round(y_max_SX) , round(x_min_SX) , round(x_max_SX))
    
    except :
        
        return None, None
        
def checkCorrectPositionStivaleSx(pose_landmarks, height, width) :
    #### Logica: 0) No girato di spalle!
    ###          1) il piede deve essere presente nell'immagine (presence)
    ####    (no) 2) il piede deve essere visibile, non deve avere niente davanti (visibility)
    ####         3) se i piedi sono "accavallati", quello davanti deve essere il dx
    ####         3.1) non devono stare proprio attaccati

    # Coordinate punti dei piedi
    caviglia_piede_SX = pose_landmarks[27]
    tallone_piede_SX = pose_landmarks[29]
    punta_piede_SX = pose_landmarks[31]
    tallone_piede_DX = pose_landmarks[30]
    punta_piede_DX = pose_landmarks[32]
    coord_spallaDx = pose_landmarks[12]
    coord_spallaSx = pose_landmarks[11]
    
    # 0)
    if ((coord_spallaSx.x * width - coord_spallaDx.x * width) < (-150)) :
        return False
    
    # 1) Presence
    if (caviglia_piede_SX.presence < 0.9 or
        tallone_piede_SX.presence < 0.9 or
        punta_piede_SX.presence < 0.9 ) :
            return False
        
    # 3) Accavallamento
    x_min_DX = min(tallone_piede_DX.x*width, punta_piede_DX.x*width)
    x_max_DX = max(tallone_piede_DX.x*width, punta_piede_DX.x*width) 
    x_min_SX = min(tallone_piede_SX.x*width, punta_piede_SX.x*width)
    x_max_SX = max(tallone_piede_SX.x*width, punta_piede_SX.x*width)
    
    # Piede sinistro "dentro" dal destro (sbaglito!)
    if ( ((x_min_SX < x_max_DX) and (x_min_SX > x_min_DX)) or ((x_max_SX < x_max_DX) and (x_max_SX > x_min_DX)) ):
        # Quello davanti deve essere sinistro (logica al contrario, sul False)
        if (punta_piede_DX.z < punta_piede_SX.z) :
            return False
        
    # 3.1) Non troppo vicini
    if (
        (abs(tallone_piede_DX.z - tallone_piede_SX.z) < 0.2 ) and
        (abs(tallone_piede_DX.x*width - tallone_piede_SX.x*width) < 40 ) ) :
        return False

    return True

def crop_stivali_dx_from_image(detection_result, image, return_coord=False) :

    # Prendo la lista dei punti del corpo rilevati ([0] perchè è la prima/unica persona)
    pose_landmarks_list = detection_result.pose_landmarks
    pose_landmarks = pose_landmarks_list[0]
    
    # Height and width image
    height = image.shape[0]
    width = image.shape[1]
    
    if (not checkCorrectPositionStivaleDx(pose_landmarks, height, width)):
        return None, None
       
    try :
         
        # Coordinate punti dei piedi SX
        caviglia_piede_DX = pose_landmarks[28]
        tallone_piede_DX = pose_landmarks[30]
        punta_piede_DX = pose_landmarks[32]
        
        x_min_DX = max(0, min(tallone_piede_DX.x*width, punta_piede_DX.x*width) - 60)
        x_max_DX = min(width, max(tallone_piede_DX.x*width, punta_piede_DX.x*width) + 60)
        y_min_DX = max(0, caviglia_piede_DX.y*height - 100)
        y_max_DX = min(height, punta_piede_DX.y*height + 100)
        
        crop_stivale_DX = image[round(y_min_DX) : round(y_max_DX) , round(x_min_DX) : round(x_max_DX)]
        
        return crop_stivale_DX, (round(y_min_DX) , round(y_max_DX) , round(x_min_DX) , round(x_max_DX))
    
    except :
        
        return None, None
    
def checkCorrectPositionStivaleDx(pose_landmarks, height, width) :
    #### Logica: 0) No girato di spalle!
    ####         1) il piede deve essere presente nell'immagine (presence)
    ####    (no) 2) il piede deve essere visibile, non deve avere niente davanti (visibility)
    ####         3) se i piedi sono "accavallati", quello davanti deve essere il dx
    ####         3.1) non devono stare proprio attaccati
    
    # Coordinate punti dei piedi
    caviglia_piede_DX = pose_landmarks[28]
    tallone_piede_DX = pose_landmarks[30]
    punta_piede_DX = pose_landmarks[32]
    tallone_piede_SX = pose_landmarks[29]
    punta_piede_SX = pose_landmarks[31]
    coord_spallaDx = pose_landmarks[12]
    coord_spallaSx = pose_landmarks[11]
    
    # 0)
    if ((coord_spallaSx.x * width - coord_spallaDx.x * width) < (-150)) :
        return False
    
    # 1) Presence
    if (caviglia_piede_DX.presence < 0.9 or
        tallone_piede_DX.presence < 0.9 or
        punta_piede_DX.presence < 0.9 ) :
            return False

    # 3) Accavallamento
    x_min_DX = min(tallone_piede_DX.x*width, punta_piede_DX.x*width)
    x_max_DX = max(tallone_piede_DX.x*width, punta_piede_DX.x*width) 
    x_min_SX = min(tallone_piede_SX.x*width, punta_piede_SX.x*width)
    x_max_SX = max(tallone_piede_SX.x*width, punta_piede_SX.x*width)
    
    # Piede destro "dentro" al sinistro (sbaglito!)
    if ( ((x_min_DX < x_max_SX) and (x_min_DX > x_min_SX)) or ((x_max_DX < x_max_SX) and (x_max_DX > x_min_SX)) ):
        # Quello davanti deve essere destro (logica al contrario, sul False)
        if (punta_piede_DX.z > punta_piede_SX.z) :
            return False
        
    # 3.1) Non troppo vicini
    if (
        (abs(tallone_piede_DX.z - tallone_piede_SX.z) < 0.2 ) and
        (abs(tallone_piede_DX.x*width - tallone_piede_SX.x*width) < 40 ) ) :
        return False
    
    return True


