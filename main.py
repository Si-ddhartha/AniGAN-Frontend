import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import os
current_directory = os.path.dirname(os.path.realpath(__file__))

generator = load_model('generator.h5')

app = FastAPI()
app.mount("/static", StaticFiles(directory=current_directory), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


def generate_image():
    image = generator.predict(tf.random.uniform((1, 128)), verbose=0)

    image = image.reshape((64, 64, 3)) 

    return image

@app.get('/generate')
def return_generated_image():
    image = generate_image()

    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_bytes = cv2.imencode(".png", image)[1].tobytes()
    return Response(content=image_bytes, media_type="image/png")

@app.get('/')
def get_index():
    index_html_path = os.path.join(current_directory, "index.html")
    return FileResponse(index_html_path)


# if __name__ == "__main__":
#     import uvicorn

#     # Run the FastAPI app using Uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
