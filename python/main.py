# Importa as bibliotecas necessárias
import serial  # Para comunicação serial com dispositivos externos
import cv2  # Para processamento de imagens e vídeos
import mediapipe as mp  # Para detecção e rastreamento de mãos
import math  # Para funções matemáticas avançadas

# Configuração das opções do programa
write_video = False  # Define se o vídeo deve ser salvo em um arquivo (True para salvar, False para não salvar)
debug = False  # Define se o modo de depuração está ativado

# Se o modo de depuração não estiver ativado, inicializa a comunicação serial
if not debug:
    ser = serial.Serial('/dev/ttyUSB0', 115200)  # Abre a porta serial '/dev/ttyUSB0' com uma taxa de 115200 bauds

# Ajustes de posição e limites para os servos e outros parâmetros
x_min = 0  # Limite mínimo para o ângulo do servo no eixo x
x_mid = 75  # Valor central para o ângulo do servo no eixo x
x_max = 150  # Limite máximo para o ângulo do servo no eixo x

y_min = 0  # Limite mínimo para o ângulo do servo no eixo y
y_mid = 90  # Valor central para o ângulo do servo no eixo y
y_max = 180  # Limite máximo para o ângulo do servo no eixo y

z_min = 10  # Limite mínimo para o ângulo do servo no eixo z
z_mid = 90  # Valor central para o ângulo do servo no eixo z
z_max = 180  # Limite máximo para o ângulo do servo no eixo z

palm_angle_min = -50  # Limite mínimo para o ângulo da palma
palm_angle_mid = 20  # Valor central para o ângulo da palma

wrist_y_min = 0.3  # Limite mínimo para a posição y do pulso
wrist_y_max = 0.9  # Limite máximo para a posição y do pulso

palm_size_min = 0.1  # Tamanho mínimo da palma
palm_size_max = 0.3  # Tamanho máximo da palma

claw_open_angle = 60  # Ângulo do servo para a garra aberta
claw_close_angle = 0  # Ângulo do servo para a garra fechada

# Inicializa a lista de ângulos dos servos com os valores centrais e a garra aberta
servo_angle = [x_mid, y_mid, z_mid, claw_open_angle]  # [x, y, z, garra]
prev_servo_angle = servo_angle  # Guarda os ângulos dos servos da iteração anterior para suavizar a transição
fist_threshold = 7  # Limite para determinar se a mão está fechada

# Inicializa o MediaPipe para desenhar as mãos e rastreá-las
mp_drawing = mp.solutions.drawing_utils  # Ferramentas de desenho para desenhar as mãos detectadas
mp_drawing_styles = mp.solutions.drawing_styles  # Estilos padrão para o desenho das mãos e conexões
mp_hands = mp.solutions.hands  # Solução do MediaPipe para rastreamento de mãos

# Inicializa a captura de vídeo da câmera
cap = cv2.VideoCapture(0)  # Captura o vídeo da câmera padrão (índice 0)

# Configura o gravador de vídeo, se necessário
if write_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define o codec para gravação de vídeo
    out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))  # Inicializa o escritor de vídeo para salvar em 'output.avi'

# Função lambda para garantir que o valor esteja dentro de um intervalo
clamp = lambda n, minn, maxn: max(min(maxn, n), minn)  # Garante que n esteja entre minn e maxn

# Função lambda para mapear um valor de um intervalo para outro
map_range = lambda x, in_min, in_max, out_min, out_max: ((x - in_min) * (out_max - out_min) / (in_max - in_min)) + out_min

# Função para verificar se a mão está fechada
def is_fist(hand_landmarks,    palm_size):
    """
    Verifica se a mão está fechada com base na posição dos marcos da mão e no tamanho da palma.
    
    :param hand_landmarks: Landmarks da mão detectados pelo MediaPipe.
    :param palm_size: Tamanho da palma da mão.
    :return: True se a mão estiver fechada, False caso contrário.
    """
    distance_sum = 0
    WRIST = hand_landmarks.landmark[0]  # Obtém o marco do pulso
    
    # Calcula a soma das distâncias entre o pulso e os outros marcos das pontas dos dedos
    for i in [7, 8, 11, 12, 15, 16, 19, 20]:
        distance_sum += ((WRIST.x - hand_landmarks.landmark[i].x) ** 2 +
                         (WRIST.y - hand_landmarks.landmark[i].y) ** 2 +
                         (WRIST.z - hand_landmarks.landmark[i].z) ** 2) ** 0.5
    
    # Verifica se a soma das distâncias dividida pelo tamanho da palma é menor que o limiar de "punho fechado"
    return distance_sum / palm_size < fist_threshold

def landmark_to_servo_angle(hand_landmarks):
    """
    Converte os marcos da mão em ângulos para os servos com base na posição da mão.
    
    :param hand_landmarks: Landmarks da mão detectados pelo MediaPipe.
    :return: Lista de ângulos dos servos [x, y, z, garra].
    """
    servo_angle = [x_mid, y_mid, z_mid, claw_open_angle]  # Inicializa os ângulos dos servos
    WRIST = hand_landmarks.landmark[0]  # Obtém o marco do pulso
    INDEX_FINGER_MCP = hand_landmarks.landmark[5]  # Obtém o marco da articulação da base do dedo indicador
    
    # Calcula o tamanho da palma
    palm_size = ((WRIST.x - INDEX_FINGER_MCP.x) ** 2 +
                 (WRIST.y - INDEX_FINGER_MCP.y) ** 2 +
                 (WRIST.z - INDEX_FINGER_MCP.z) ** 2) ** 0.5

    # Ajusta o ângulo da garra com base em se a mão está fechada ou não
    if is_fist(hand_landmarks, palm_size):
        servo_angle[3] = claw_close_angle  # Define a garra como fechada
    else:
        servo_angle[3] = claw_open_angle  # Define a garra como aberta

    # Calcula o ângulo x
    delta_x = WRIST.x - INDEX_FINGER_MCP.x
    delta_z = WRIST.z - INDEX_FINGER_MCP.z
    distance = math.sqrt(delta_x**2 + delta_z**2)  # Calcula a distância entre os pontos

    if distance != 0:
        angle = math.atan2(delta_z, delta_x)  # Calcula o ângulo em radianos
        angle = math.degrees(angle)  # Converte o ângulo para graus
        angle = clamp(angle, palm_angle_min, palm_angle_mid)  # Limita o ângulo para estar dentro dos valores permitidos
        servo_angle[0] = map_range(angle, palm_angle_min, palm_angle_mid, x_max, x_min)  # Mapeia o ângulo para o intervalo do servo
    else:
        servo_angle[0] = x_mid  # Define o ângulo x para o valor central se a distância for zero

    # Calcula o ângulo y com base na posição y do pulso
    wrist_y = clamp(WRIST.y, wrist_y_min, wrist_y_max)  # Limita a posição y do pulso
    servo_angle[1] = map_range(wrist_y, wrist_y_min, wrist_y_max, y_max, y_min)  # Mapeia a posição y para o intervalo do servo

    # Calcula o ângulo z com base no tamanho da palma
    palm_size = clamp(palm_size, palm_size_min, palm_size_max)  # Limita o tamanho da palma
    servo_angle[2] = map_range(palm_size, palm_size_min, palm_size_max, z_max, z_min)  # Mapeia o tamanho da palma para o intervalo do servo

    # Converte os ângulos dos servos para inteiros
    servo_angle = [int(i) for i in servo_angle]

    return servo_angle

def smooth_transition(current_angles, target_angles, smooth_factor=0.1):
    """
    Suaviza a transição dos ângulos dos servos entre o ângulo atual e o alvo.
    
    :param current_angles: Ângulos atuais dos servos.
    :param target_angles: Ângulos de destino para os servos.
    :param smooth_factor: Fator de suavização (quanto menor, mais suave a transição).
    :return: Lista de ângulos suavizados dos servos.
    """
    return [int(current + (target - current) * smooth_factor) for current, target in zip(current_angles, target_angles)]

smooth_factor = 0.1  # Fator de suavização para a transição dos ângulos dos servos

# Inicializa o módulo de rastreamento de mãos do MediaPipe
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():  # Continua a captura de vídeo enquanto a câmera estiver aberta
        success, image = cap.read()  # Captura um frame da câmera
        if not success:  # Verifica se a captura foi bem-sucedida
            print("Ignoring empty camera frame.")  # Exibe uma mensagem se o frame estiver vazio
            continue

        image.flags.writeable = False  # Desativa a escrita na imagem para melhorar o desempenho
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte a imagem de BGR para RGB
        results = hands.process(image)  # Processa a imagem para detectar as mãos

        image.flags.writeable = True  # Reativa a escrita na imagem
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Converte a imagem de volta para BGR

        if results.multi_hand_landmarks:  # Se houver marcos das mãos detectados
            if len(results.multi_hand_landmarks) == 1:  # Se houver exatamente uma mão detectada
                hand_landmarks = results.multi_hand_landmarks[0]  # Obtém os marcos da primeira mão detectada
                new_servo_angle = landmark_to_servo_angle(hand_landmarks)  # Converte os marcos da mão em ângulos de servo

                if new_servo_angle != prev_servo_angle:  # Se os ângulos dos servos mudaram
                    smooth_angle = smooth_transition(prev_servo_angle, new_servo_angle, smooth_factor)  # Suaviza a transição dos ângulos
                    print("Servo angle: ", smooth_angle)  # Imprime os ângulos dos servos
                    prev_servo_angle = smooth_angle  # Atualiza o ângulo dos servos anterior
                    if not debug:  # Se o modo de depuração não estiver ativado
                        ser.write(bytearray(smooth_angle))  # Envia os ângulos dos servos para o dispositivo através da comunicação serial
            else:
                print("More than one hand detected")  # Imprime uma mensagem se mais de uma mão for detectada
                
            # Desenha os marcos das mãos e suas conexões na imagem
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        image = cv2.flip(image, 1)  # Inverte a imagem horizontalmente
        cv2.putText(image, str(prev_servo_angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # Adiciona o texto com os ângulos dos servos na imagem
        cv2.imshow('MediaPipe Hands', image)  # Exibe a imagem com os marcos das mãos desenhados

        if write_video:  # Se a gravação de vídeo estiver ativada
            out.write(image)  # Salva o frame atual no arquivo de vídeo
        if cv2.waitKey(5) & 0xFF == 27:  # Verifica se a tecla ESC foi pressionada
            if write_video:
                out.release()  # Libera o escritor de vídeo
            break

cap.release()  # Libera a captura de vídeo

