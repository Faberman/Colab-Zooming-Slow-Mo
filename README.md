# Colab-Zooming-Slow-Mo (CVPR-2020)

By [Xiaoyu Xiang<sup>*</sup>](https://engineering.purdue.edu/people/xiaoyu.xiang.1), [Yapeng Tian<sup>*</sup>](http://yapengtian.org/), [Yulun Zhang](http://yulunzhang.com/), [Yun Fu](http://www1.ece.neu.edu/~yunfu/), [Jan P. Allebach<sup>+</sup>](https://engineering.purdue.edu/~allebach/), [Chenliang Xu<sup>+</sup>](https://www.cs.rochester.edu/~cxu22/) (<sup>*</sup> equal contributions, <sup>+</sup> equal advising)

This is a Google Colab Notebook for the official Pytorch implementation of *Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time Video Super-Resolution*. 

## Simply download `Colab-Zooming-Slow-Mo.ipynb` and open it inside your Google Drive or click [here](https://colab.research.google.com/github/styler00dollar/Colab-Zooming-Slow-Mo/blob/master/Colab-Zooming-Slow-Mo.ipynb) and copy the file with "File > Save a copy to Drive..." into your Google Drive. 

## Important information

- If you can't open `Colab-Zooming-Slow-Mo.ipynb` inside your Google Drive, try this [colab link](https://colab.research.google.com/github/styler00dollar/Colab-Zooming-Slow-Mo/blob/master/Colab-Zooming-Slow-Mo.ipynb) and save it to your Google Drive. The "open in Colab"-button can be missing in Google Drive, if that person never used Colab.
- Google Colab does assign a random GPU. It depends on luck.
- The Google Colab VM does have a maximum session length of 12 hours. Additionally there is a 30 minute timeout if you leave colab. The VM will be deleted after these timeouts.


#### [Paper](https://arxiv.org/abs/2002.11616) | [Demo (YouTube)](https://youtu.be/8mgD8JxBOus) | [1-min teaser (YouTube)](https://www.youtube.com/watch?v=C1o85AXUNl8) | [1-min teaser (Bilibili)](https://www.bilibili.com/video/BV1GK4y1t7nb/)

Input-> [![GIF demo](dump/demo720.gif)](https://youtu.be/8mgD8JxBOus) <- Output

## Updates
- **Our paper will have Q&A session on June 16, 2-4 am/pm PDT. Welcome to come and ask questions!**
- 2020.3.13 Add meta-info of datasets used in this paper
- 2020.3.11 Add new function: video converter
- 2020.3.10: Upload the complete code and pretrained models

## Introduction
The repository contains the entire project (including all the preprocessing) for one-stage space-time video super-resolution with Zooming Slow-Mo. 

Zooming Slow-Mo is a recently proposed joint video frame interpolation (VFI) and video super-resolution (VSR) method, which directly synthesizes an HR slow-motion video from an LFR, LR video. It is going to be published in [CVPR 2020](http://cvpr2020.thecvf.com/). The most up-to-date paper with supplementary materials can be found at [arXiv](https://arxiv.org/abs/2002.11616). 

In Zooming Slow-Mo, we firstly temporally interpolate features of the missing LR frame by the proposed feature temporal interpolation network. Then, we propose a deformable ConvLSTM to align and aggregate temporal information simultaneously. Finally, a deep reconstruction network is adopted to predict HR slow-motion video frames. If our proposed architectures also help your research, please consider citing our paper.

Zooming Slow-Mo achieves state-of-the-art performance by PSNR and SSIM in Vid4, Vimeo test sets.

![framework](./dump/framework.png)

## Citations
If you find the code helpful in your resarch or work, please cite the following papers.
```BibTex
@InProceedings{xiang2020zooming,
  author = {Xiang, Xiaoyu and Tian, Yapeng and Zhang, Yulun and Fu, Yun and Allebach, Jan P. and Xu, Chenliang},
  title = {Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time Video Super-Resolution},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={3370--3379},
  month = {June},
  year = {2020}
}

@InProceedings{tian2018tdan,
  author={Yapeng Tian, Yulun Zhang, Yun Fu, and Chenliang Xu},
  title={TDAN: Temporally Deformable Alignment Network for Video Super-Resolution},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={3360--3369},
  month = {June},
  year = {2020}
}

@InProceedings{wang2019edvr,
  author    = {Wang, Xintao and Chan, Kelvin C.K. and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title     = {EDVR: Video restoration with enhanced deformable convolutional networks},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  month     = {June},
  year      = {2019},
}
```

## Contact
[Xiaoyu Xiang](https://engineering.purdue.edu/people/xiaoyu.xiang.1) and [Yapeng Tian](http://yapengtian.org/). 

You can also leave your questions as issues in the repository. We will be glad to answer them.

## License
This project is released under the [GNU General Public License v3.0](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020/blob/master/LICENSE).

## Acknowledgments
Our code is inspired by [TDAN-VSR](https://github.com/YapengTian/TDAN-VSR) and [EDVR](https://github.com/xinntao/EDVR).
