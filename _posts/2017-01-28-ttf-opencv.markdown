---
layout: post
title:  "OpenCV module for drawing UTF-8 strings with freetype2"
date:   2017-01-28 16:50:00 -0700
categories: Misc
---

Thankfully a new module has been added to OpenCV that makes rendering text using the fonts supported by FreeType (including TrueType) onto images very easy. [This page](http://docs.opencv.org/3.2.0/d4/dfc/group__freetype.html) documents this module. In Python that means there is no need to use Cairo or PIL. The following code shows how this modules can be used in Python.

```python
import cv2
import numpy as np

img = np.zeros((100, 300, 3), dtype=np.uint8)

ft = cv2.freetype.createFreeType2()
ft.loadFontData(fontFileName='Ubuntu-R.ttf',
                id=0)
ft.putText(img=img,
           text='Quick Fox',
           org=(15, 70),
           fontHeight=60,
           color=(255,  255, 255),
           thickness=-1,
           line_type=cv2.LINE_AA,
           bottomLeftOrigin=True)

cv2.imwrite('image.png', img)
```

The PNG image should like this: 
![Image generated with OpenCV and text using the Ubuntu font](https://github.com/fireant/fireant.github.io/raw/master/assets/opencv_ttf_quick_fox.png)

