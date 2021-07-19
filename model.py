from typing import Union, Any


class GFile:
    def __init__(self):
        self.kind = None
        self.id = None
        self.name = None
        self.mimeType = None

    def getProperties(self):
        return {'kind': self.kind, 'id': self.id,
                'name': self.name, 'mimeType': self.mimeType}

    def setProperties(self, kind, id, name, mimeType):
        self.id = id
        self.kind = kind
        self.name = name
        self.mimeType = mimeType

    def setPropertiesFromDict(self, fileDict: dict):
        values = tuple(fileDict.values())
        self.setProperties(*values)

    def getId(self):
        return self.id

    def getKind(self):
        return self.kind

    def getName(self):
        return self.name

    def getMimeType(self):
        return self.mimeType

    def setName(self, name):
        self.name = name

    def setMimeType(self, mimeType):
        self.mimeType = mimeType

    def setId(self, id):
        self.id = id

    def setKind(self, kind):
        self.kind = kind

    def __str__(self):
        return "id: %s, kind: %s, mimeType: %s, name: %s" % (self.id, self.kind, self.mimeType, self.name)


class Face:
    def __init__(self):
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.mask = False

    def setCoordinates(self, x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def setMask(self, mask: bool = False):
        self.mask = mask

    def setAll(self, x1: Union[int, str], y1: Union[int, str], x2: Union[int, str], y2: Union[int, str],
               mask: [str, bool, int]):
        self.x1 = int(x1)
        self.x2 = int(x2)
        self.y1 = int(y1)
        self.y2 = int(y2)
        if isinstance(mask, str):
            mask = int(mask)

        if isinstance(mask, int):
            mask = True if mask == 1 else False

        self.mask = mask

    def isMasked(self):
        return self.mask

    def getCoordinates(self):
        return self.x1, self.y1, self.x2, self.y2


class Image:
    def __init__(self):
        self.path = None
        self.face = []

    def setFaces(self, *face: Union[Any]):
        for f in face:
            fx = tuple(f.split(','))
            face = Face()
            face.setAll(*fx)
            self.setFace(face)

    def setFace(self, face: Union[Face, None]):
        self.face.append(face)

    def setPath(self, path: Union[str, None]):
        self.path = path

    def getParams(self):
        return {
            'path': self.path,
            'face': self.face
        }

    def __str__(self):
        print('path', self.path)
        print('face', self.face)
        return ""

    def countFace(self):
        return len(self.face)

    def getFaces(self):
        return self.face

    def getPath(self):
        return self.path
