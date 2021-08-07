# Classification of Masked and Unmasked Faces using Transfer Learning Concept.

### Principle Dataset Link:
Uploaded by Sunil Singh (sunil32123singh@gmail.com) - [https://drive.google.com/drive/folders/1pAxEBmfYLoVtZQlBT3doxmesAO7n3ES1](https://drive.google.com/drive/folders/1pAxEBmfYLoVtZQlBT3doxmesAO7n3ES1) which is publicly available. From this database we separate faces.

### Obtaining & Configuring the dataset (from Google Drive to Computer)

##### Steps:

1. **Use:** `pip install google-api-python-client`.
2. **Please refer to the Google Documentation on OAuth 2.0 & Drive API v3.**
3. **Please enter:** `http://localhost:8080/` as a `redirect_uri` in the cloud console of Google for your Client ID & Secret Pair.
4. **Please set up:** the OAuth 2.0 screen with one of the sensitive scopes as `./auth/drive` for full access to **Google Drive API**.

After registering the client keys please do not forget to download and save `client_secrets.json`, at the same path as your program. That section is uploaded to [https://github.com/formula21/face-mask/blob/main/croppie.py](https://github.com/formula21/face-mask/blob/main/croppie.py).

##### Other notes:

1. The `imagebuffer` once stored as a file, stays on your computer, unless you define the `FLAG_UPLOAD = true` and `FLAG_UPLOAD_PARENT_ID = [None, None] or [str, str]`.
  - If either `None` is defined,  a **new folder** with the names `with_mask` and `without_mask` are created or the id's are used to upload. If a failure of finding the directories in drive, the program terminates with an Exception.
  - If either or both are found, we will send the buffer immediately to upload.
2. By default, the file is supposed to be also saved locally, however this can be omitted by defining `FLAG_DOWNLOAD_AND_SAVE = false`.

### Google Drive Authorization

We are authorizing Google Colab with Google Drive for us to get access to our dataset. This authorization is native to Google's Documentation.

**Please note:** You need to change the variable `dir` below to the appropriate path. If you are using a "Shared With Me" folder please set up a "Add Shortcut to Drive", to set shortcut and get easy access to &ldquo;My Drive&rdquo;.


### Jupyter Notebook & Google Colab

For training the model, the system did not have enough memory. We had to write a modified code on Google Colab, only to train & test the model.

### License

`The project is licensed with an open source license from MIT. A copy of this license should be there exported with every copy of the library you clone or download. However the license reads as follows:`


> Copyright (c) 2021 Memo Karpa <memokarpa001@gmail.com>, Anweshan Roy Chowdhury <anweshanrc15@gmail.com>, Sayon Roy <roysayon2000@gmail.com>,  Subhajit Majumdar <subhajitmajumdar56@gmail.com>.

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:

> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.

> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
