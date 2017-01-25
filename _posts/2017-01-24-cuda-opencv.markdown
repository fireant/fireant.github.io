---
layout: post
title:  "Enable CUDA and CUBLAS in OpenCV"
date:   2016-05-21 12:57:30 -0700
categories: Misc
---

To enable CUDA and CUBLAS in OpenCV at compile time define these params when calling `cmake`, `-D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1`.
I have figure out later how to tell it to use `NVCUVID`. 

