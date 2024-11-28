from fastapi import FastAPI, WebSocket
from ultralytics import YOLO
import numpy as np
import cv2
import uvicorn
import torch
import json
import asyncio
import websockets

# Servidor (GPU PC)
app = FastAPI()

model = YOLO('runs\\detect\\train3\\weights\\best.pt')
if torch.cuda.is_available():
    model = model.cuda()
    print("Usando GPU:", torch.cuda.get_device_name(0))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Cliente conectado")
    
    while True:
        try:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Puedes ajustar parámetros de inferencia si lo necesitas
            results = model(frame, conf=0.6)  # Ajusta el umbral de confianza según necesites
            
            annotated_frame = results[0].plot()
            cv2.imshow('Detecciones', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            await websocket.send_text("ok")
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    cv2.destroyAllWindows()
    print("Cliente desconectado")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)