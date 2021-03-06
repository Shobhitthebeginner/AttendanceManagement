import pyrebase
import os
import cv2
from PIL import Image
import requests
from io import BytesIO
import requests
from PIL import Image
from io import BytesIO
import numpy as np


# ---------------- Firestore -----------------------------------#

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import cv2
import urllib
import urllib.request
import numpy as np

config = {
    "type": "service_account",
    "project_id": "image-23704",
    "private_key_id": "3356ea3be1eacd942f2e46b16924c2fc1d821dd1",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDI1Nxc9eHSPeYB\nSNfMWzkKuEQU6YknVm6OtT8zghD84sZ3GHFmGs2QsHRjTdxj6fSwRoYnPnemUB/M\npDjDotKNjbni4KFkqB69gQhTWxJeREZtjsLfWhtxCaSEH0bbzMWFpEkxl5ds8FiR\npq9f5p4gGs1cMTlURlsEzixE/Ue1tnf9+pBKerqy7OzAWXLgbPaqHv6BMPqBwKHS\nwqftXlBy6idxRM+4BV5gzHD9ASdms1rvsOzOfog6SURmbgEtuURzr6s5AUOwutut\npmPu3sjeR6aPLzB4e6H30MzIYT60cxQuFAYyBSJ+8dRQUoqsS6qKwXULwl85epvY\nmC8DK0jlAgMBAAECggEABYYWnwQSkxjlWMyyrKn0dg8vvcukD5iPOl9rGm6c1Zp4\npoKqxBJ7QUsKxc+FmJtkTqDDjRz6CXFiqsoaeXSnH07+7JSdtFTQYcUaBFrHzoWe\nJ8BnXp0dBcbfnqE47pzOaESASBN6J6bhW2y+82G/OkH2uwiWXaZnNZ5TkZZWfp+2\nePD5vbBqi56JuTkZeuu82TVlmMsRoKnnNP6YmZ0nfv6Q9yTHKT3c1WbZCJvOg8sP\nRuo7ZPsAqPWcBKO2F6N9VGij7xWV84gqaAD/k9s/337kvr1c9+cfWmkI7TY19g3h\n7b2ltbBBQ/PAS2ZcHfAsNhC3ub5l34fbJRaaJOeo8wKBgQD9R5ZksYOFHo8GMt6K\n6RWYoyY412cZo82JvCBg8lF/cuQYaK+2IU5yB3+7LdopB477XpAkijoTopecnpJx\ngUWImdqCnIZv1yt/bDWzVWLs6y2VLSgbJJgt9XLAiJGdtBo/55996JrjTbQdUlwB\nvjXorJhW7rbifjD8ESucV3jerwKBgQDK/RAdt74gbYM3oLYGE4GIVno46eRFrDW9\nF4wPuV9Xt8KJ7mW1EbIM/XGmM+LK3agoWm4k0bNVfau7S/XQ7x+Z4BOISVMPyzpR\newwm6IwOAVFpdD9kmmqyAD0gjmKfh4F0OVYDvoF+F4TZwFBjbELyGNgC77V2p2CE\nNh8uiUyWqwKBgAwxbREN5qn67aG7wzDmxa5idE2aORFn7FYsI1bnc3ryOf7e006u\nTct5hvGo5G7DOWPqin/n06HsWuYkUCJ8ua840Ocmx+YMcsCgofkvLCMBs2ESGnMs\nENNtlIemS3RPHlBjQy9ZilNVA03CEEHZOVkpLfBJb655qrwHy5SsNVprAoGBALRn\n8VjYIuwjKInaFayUzXzkjr/ib/TUNvaV5O9cqzYEpat863vf/ES7Q7SZTKlMEtW6\neUXT8fS7OlO+EPzeaVGS6wknUeEpl+0u1QAHkeIonbiBjo3VB5qnx6wVn+V0w0MO\najntqJzuPi5hU5DpeR49ok4JyVdpLsiSaWgsspr9AoGAIsFvGL3HCv73UbUyybWX\n9DLEc8M2bNgOFvOWWRj7NsKTm2zagutz/6DcVgXhI/Uct9xn1FDTblAX5IOwMkXz\noFveCJBeVpQcu11J0M7naEg6buv0mpX7DscAZMT1nP9rnEXwyFDuX3wYPvK0PMHH\nRUotg0rctiv9ps4Mw7/NR4w=\n-----END PRIVATE KEY-----\n",
    "client_email": "firebase-adminsdk-a3tb3@image-23704.iam.gserviceaccount.com",
    "client_id": "116186165584908361579",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-a3tb3%40image-23704"
                            ".iam.gserviceaccount.com "
}

cred = credentials.Certificate(config)
firebase_admin.initialize_app(cred)
db = firestore.client()
user = db.collection('image').document('images').get().to_dict()
Name=list(user.keys())
Urls=list(user.values())
images=[] # containing matrix
classNames=[] # containing names
for i in range(len(Urls)):
    req = urllib.request.urlopen(Urls[i])
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)  # 'Load it as it is'
    # cv2.imshow('lalala', img)
    # cv2.waitKey(0)
    classNames.append(img)
    images.append(Name[i])

# images=[]  # containing images names with their extensions
# classNames=[]  # containing matrix
#
#


# print(classNames)
# print(images)
# # encoding process
# def findEncodings(images):
#     encodeList=[]
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode=face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList
# email = "youremail@gmail.com"
# password = "yourpassword"
#
# # create new user
# db.collection('images').document(email).set({'email': email, 'password': password})
#
# # add new data to the existing user
# db.collection('images').document(email).update({'name': "abcd"})
#
# # get data
# user = db.collection('images').document(email).get().to_dict()
#
# # check user exsist or not
# if user:
#     print("user exsist")
#     if(email==user['email'] and password==user['password']):
#         print("login sucessfully")
#     else:
#         print("wrong password")
# else:
#     print('user not exsist')




'''


# # Create user
# email = "youremail@gmail.com"
# password = "yourpassword"
# auth.create_user_with_email_and_password(email, password)

# Sign in
user = auth.sign_in_with_email_and_password(email, password)
# print(user)
# print(user['idToken'])

# Get account information
info = auth.get_account_info(user['idToken'])
print(info)

# Verify email
auth.send_email_verification(user['idToken'])

# Reset email
auth.send_password_reset_email(email)

# Delete account
auth.delete_user_account(user['idToken'])

#  Before 1 hour expiry
user = auth.refresh(user['refreshToken'])
print(user['idToken'])
'''
