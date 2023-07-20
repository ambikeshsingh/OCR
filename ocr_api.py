# import required libraries

import cx_Oracle
import ssl
import cv2
# from uvicorn import Config, SSLConfig
import numpy as np
import easyocr
from io import BytesIO
import getpass
import cv2
# import oracledb
import cx_Oracle
import datetime
import pandas as pd
from PIL import Image
import base64
import tkinter as tk
from PIL import Image as im
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, Form
import io
import re
import cv2
import base64
import imutils
from fastapi import Request, Body
import uvicorn
from pydantic import BaseModel
import json
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

cx_Oracle.init_oracle_client(lib_dir="C:\Oracle instant\instantclient_21_9")
# class Face(BaseModel):
#     Employee_ID: str
# from flask import Flask, request
app = FastAPI()
dsn = cx_Oracle.makedsn(
    'localhost',
    '1521',
    service_name='orcl'
)
conn = cx_Oracle.connect(
    user='ambikesh_singh',
    password='MyPassword',
    dsn=dsn
)


# c = conn.cursor()

@app.post("/OCR")
async def ocr(Consumer_id: str, Ocr_image: UploadFile = File(...)):
    # data = jsonable_encoder(data)
    # dsn = cx_Oracle.makedsn(
    #     'localhost',
    #     '1521',
    #     service_name='orcl')
    # conn = cx_Oracle.connect(
    #     user='SYSTEM',
    #     password='SYSTEM',
    #     dsn=dsn)

    c = conn.cursor()
    contents = await Ocr_image.read()

    # Check if the consumer ID exists in the database
    c.execute('SELECT COUNT(*) FROM "OCR" WHERE "CONSUMER_ID" =' + Consumer_id + '')
    count = c.fetchone()[0]

    if count > 0:
        # Update the image for the existing consumer ID
        query = "UPDATE OCR SET OCR_IMAGE = :blobdata WHERE CONSUMER_ID = :Consumer_id"
        c.execute(query, {"blobdata": contents, "Consumer_id": Consumer_id})
        # c.execute("INSERT INTO FACE_RECOGNITION (EMPLOYEE_ID, CURRENT_IMAGE) VALUES (:1, :2)", (Employee_ID, contents))

        # c.execute('UPDATE OCR_DATA SET "OCR_IMAGE" = :blobdata  WHERE "CONSUMER_ID" =' + Consumer_id + '')
        conn.commit()
        # print("Image updated successfully.")
        c.execute('SELECT "CONSUMER_ID" from "OCR" where "CONSUMER_ID" =' + Consumer_id + '')
        e_id = c.fetchall()
        i = 0
        while i < len(e_id):
            i += 1
            c = conn.cursor()
            c.execute('SELECT "OCR_IMAGE" FROM "OCR" where "CONSUMER_ID" =' + Consumer_id + '')
            result = c.fetchone()
            image_data = result[0].read()
            image = Image.open(BytesIO(image_data))

            # save the image to a file
            file_name = 'ocr_img.png'
            image.save(file_name)
            with open("ocr_img.png", "rb") as f:
                img_data = f.read()
            img = cv2.imread('ocr_img.png')
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]


            inverted_image = np.invert(thresh)

            reader = easyocr.Reader(['en'], gpu=False)
            # result = reader.readtext('D:\OCR\paint4_inverted.png', contrast_ths = 0.05, adjust_contrast = 0.7, width_ths =0.7, decoder = 'beamsearch')
            ocr_result = reader.readtext(inverted_image, detail=0, batch_size=50, adjust_contrast=0.5,
                                     allowlist='0123456789.')
            print(ocr_result)
            print(len(ocr_result))
            if(len(ocr_result)>=1):
                # result=[float(x) for x in ocr_result]
                reading=max(ocr_result,key=len)
                print(reading)

                return {"status": 200,
                    "message": "Image updated successfully.",
                    "Reading": reading}
            else:
                return {"status": 201,
                        "message": "Image not clear",
                        }



    else:
        # Insert a new record with the image and consumer ID
        c.execute('INSERT INTO "OCR" (CONSUMER_ID, OCR_IMAGE) VALUES (:id, :img)', id=Consumer_id, img=contents)
        conn.commit()
        print("Image inserted successfully.")
        c.execute('SELECT "CONSUMER_ID" from "OCR" where "CONSUMER_ID" =' + Consumer_id + '')
        e_id = c.fetchall()
        i = 0
        while i < len(e_id):
            i += 1
            c = conn.cursor()
            c.execute('SELECT "OCR_IMAGE" FROM "OCR" where "CONSUMER_ID" =' + Consumer_id + '')
            result = c.fetchone()
            image_data = result[0].read()
            image = Image.open(BytesIO(image_data))

            # save the image to a file
            file_name = 'else_ocr_img.png'
            image.save(file_name)



            with open("else_ocr_img.png", "rb") as f:
                img_data = f.read()

            img = cv2.imread('else_ocr_img.png')
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]


            inverted_image = np.invert(thresh)


            reader = easyocr.Reader(['en'], gpu=False)
            # result = reader.readtext('D:\OCR\paint4_inverted.png', contrast_ths = 0.05, adjust_contrast = 0.7, width_ths =0.7, decoder = 'beamsearch')
            ocr_result = reader.readtext(inverted_image, detail=0, batch_size=50, adjust_contrast=0.5,
                                     allowlist='0123456789.')
            print(type(ocr_result))
            if (len(ocr_result) >= 1):
                # result = [float(x) for x in ocr_result]
                reading = max(ocr_result,key=len)
                # if(len(new_result)>1):
                print(type(reading))

                return {"status": 200,
                        "message": "Image updated successfully.",
                        "Reading": reading}
            else:
                return {"status": 201,
                        "message": "Image not clear",
                        }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0')

                # ssl_keyfile='C:/Users/MDP/OCR/mpcz.key',
                # ssl_certfile='C:/Users/MDP/OCR/mpcz.crt')
