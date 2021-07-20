import io
import os
import pickle
from typing import Union, Any, List
import numpy as np
import cv2

from googleapiclient.http import MediaIoBaseDownload

from model import Image, GFile

# For google client imports
import google
import googleapiclient
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# The scopes we want to authorize.
# Please delete "token.pickle" if you are altering scope
SCOPES: List[Union[str, Any]] = ['userinfo.email', 'userinfo.profile', 'drive']
MAIN_URL: str = 'https://www.googleapis.com/auth/'


def analyzeAnnotations(file: str = 'annotations_train.txt') -> Union[List, None]:
    dataset = []
    with open(file) as fp:
        Lines = fp.readlines()
        for line in Lines:
            ln = line.strip().split()
            img = Image()
            img.setPath(ln[0].split("/")[-1])
            face = tuple(ln[1:])
            img.setFaces(*face)
            dataset.append(img)
    return dataset


def getGoogleAuthentication(port: int = 8080, browser: bool = False) \
        -> Union[None, google.oauth2.credentials.Credentials]:
    _credentials = None
    if os.path.exists('token.pickle') and not browser:
        with open('token.pickle', 'rb') as token:
            _credentials = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not _credentials or not _credentials.valid:
        if _credentials and _credentials.expired and _credentials.refresh_token:
            _credentials.refresh(Request())
        else:
            scopes = [MAIN_URL + x for x in SCOPES]
            scopes.append("openid")

            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secrets.json', scopes=scopes)

            _credentials = flow.run_local_server(port=port)

        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(_credentials, token)
    return _credentials


def getGoogleDriveService(credentials: google.oauth2.credentials.Credentials) -> \
        Union[googleapiclient.discovery.Resource, object, None]:
    service = build("drive", "v3", credentials=credentials)
    return service


def getFileList(service, q, spaces="drive", fields="nextPageToken, files(id, name, mimeType, kind)", pageSize=100):
    pageToken = None
    fileList = []

    while True:
        response = service.files().list(q=q,
                                        spaces=spaces,
                                        fields=fields,
                                        pageToken=pageToken,
                                        pageSize=pageSize).execute()

        for f in response.get('files', []):
            x = GFile()
            x.setPropertiesFromDict(f)
            fileList.append(x)

        pageToken = response.get('nextPageToken', None)

        if pageToken is None:
            break
    # Extend the list if you want the values.
    # To delete duplicates, just do list(set([]))
    # If re-orders stuff, but does not keep any duplicates
    return fileList


def getFileByName(parent: str, name: str) -> str:
    query = "'%s' in parents and trashed=false and mimeType='image/jpeg' " \
            "and name='%s'" % (parent, name)
    return query


def getAllFiles(parent: str):
    query = "'%s' in parents and trashed=false and mimeType='image/jpeg'" % parent
    return query


def downloadFile(service, gfile: [GFile]):
    request = service.files().get_media(fileId=gfile.getId())
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request, chunksize=204800)
    done = False
    try:
        # Download the data in chunks
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        return fh
    except:
        print("Something went wrong.")
        return False


def ioToImage(image_stream):
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def cropImageByCoordinates(img, obj):
    maskDir, unmaskDir, name, faces = os.path.abspath('with_mask'), os.path.abspath('without_mask'), obj.getPath(), obj.getFaces()
    for f in range(obj.countFace()):
        x1, y1, x2, y2 = faces[f].getCoordinates()
        label = "non masked" if faces[f].isMasked() else "masked"
        d = maskDir if faces[f].isMasked() else unmaskDir
        d = os.path.join(d, os.path.splitext(name)[0] + "_" + str((f + 1)) + os.path.splitext(name)[1])

        if os.path.exists(d):
            os.remove(d)

        # Cropping the image based
        # starty:endy, startx:endx
        cropImg = img[y1:y2, x1:x2]
        # Write Image
        cv2.imwrite(d, cropImg)
        if os.path.isfile(d):
            continue
        else:
            print("Face %d of file %s has been saved with filename %s and the face is %s: "
                  % ((f + 1), obj.getPath(), os.path.basename(d), label))

        # Commented Section

        # cv2.imshow(os.path.basename(d), cropImg)
        # cv2.waitKey(0)
        # cv2.destroyWindow(os.path.basename(d))


if __name__ == '__main__':
    # Parent ID (train2)
    parent = '1gbDcWb_YRzkCS3OaBA5zOdQfKr68yN8z'

    db = analyzeAnnotations()
    if not db:
        print("Dataset incorrectly analyzed. Check!!")
    credentials = getGoogleAuthentication()
    service = None
    if credentials:
        service = getGoogleDriveService(credentials)

    for img in db:
        # Making the query to Google Drive API
        query = getFileByName(parent, img.getPath())

        # noinspection PyTypeChecker
        # Pulling the FileList
        tmp = getFileList(service, query)

        if len(tmp) == 0:
            print("The file by Name %s was not found. Please check dataset at parent %s" % (img.getPath(), parent))
            continue

        # noinspection PyTypeChecker
        # Downloading the first file
        download = downloadFile(service, tmp[0])

        if download is None:
            print("The file with name %s could not be downloaded." % img.getPath())
            continue

        # TMP IMG
        tmpImg = ioToImage(download)

        # Cropping
        cropImageByCoordinates(tmpImg, img)
