import serial
import time

# Configura el puerto serial (ajusta el puerto según sea necesario)
ser = serial.Serial('COM3', 115200)  # Para Windows
# ser = serial.Serial('/dev/ttyUSB0', 115200)  # Para Linux

# Espera a que el puerto serial esté listo
time.sleep(2)

# Enviar datos al ESP32
try:
    while True:
        response = "1"
        ser.write((response + '\n').encode('utf-8'))
        time.sleep(2)
except KeyboardInterrupt:
    ser.close()
