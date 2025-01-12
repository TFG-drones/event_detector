from fastapi import FastAPI, WebSocket
from ultralytics import YOLO
import numpy as np
import cv2
import uvicorn
import torch
from uuid import uuid4

# Servidor (GPU PC)
app = FastAPI()

model = YOLO('runs\\detect\\train3\\weights\\best.pt')
if torch.cuda.is_available():
    model = model.cuda()
    print("Usando GPU:", torch.cuda.get_device_name(0))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid4())  # Generar un identificador único para el cliente
    print(f"Cliente conectado: {client_id}")
    
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            results = model(frame, conf=0.6)

            # Analizar detecciones
            detected_labels = [result.names[int(cls)] for result in results for cls in result.boxes.cls]
            print(detected_labels)
            if 'TestbedFire' in detected_labels:
                response_message = 'fire'
            else:
                response_message = 'ok'
            
            annotated_frame = results[0].plot()
            window_title = f'Detecciones - Cliente {client_id}'
            cv2.imshow(window_title, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            await websocket.send_text(response_message)
            
    except Exception as e:
        print(f"Error con cliente {client_id}: {e}")
    finally:
        cv2.destroyWindow(f'Detecciones - Cliente {client_id}')
        print(f"Cliente desconectado: {client_id}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
