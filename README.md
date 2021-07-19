# Classification of mask and non masked faces using Deep Learning.

### Dataset Link:
- Uploaded by Suhel Singh @ [https://drive.google.com/drive/folders/1pAxEBmfYLoVtZQlBT3doxmesAO7n3ES1](https://drive.google.com/drive/folders/1pAxEBmfYLoVtZQlBT3doxmesAO7n3ES1) which is publicly available

### Dataset making:
- Based on `annotation_train.txt` used python to analyze file & simultaneously called upon `Google Drive API`, to pull all images.

#### OAuth 2.0 & it's Significance:
- Registered client id and client secret [console.cloud.google.com](https://console.cloud.google.com). Follow google documentation.
- Download and rename `client_secrets.json` and place it in parent folder.

##### Procedure:
1. Goto console.cloud.google.com
2. Make a new project.
3. Register an API Key against the same. Do not forget to configure your OAuth Screen with Scopes
  - ./auth/userinfo.email
  - ./auth/userinfo.profile
  - ./auth/drive **Sensitive Scope**

#### Folder 0 & 1:
1. According to the dataset, class `0`, means all non-masked images.
2. According to the dataset, class `1`, means all masked images.

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
