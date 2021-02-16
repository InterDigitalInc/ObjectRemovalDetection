/*
 * Copyright 2021 InterDigital
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * License is avaiable on the root under license.txt and you may obtain a copy
 * of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#define STEP 0.5
#undef PALAZZOLO
#undef NON_ZERO
#define INSERTIONS

#if defined(NON_ZERO)
  #if defined(PALAZZOLO)
    #define SUFFIX "_PA_NZ"
  #else
    #define SUFFIX "_ID_NZ"
  #endif //PALAZZOLO
#else
  #if defined(PALAZZOLO)
    #define SUFFIX "_PA"
  #else
    #define SUFFIX "_ID"
  #endif //PALAZZOLO
#endif //NON_ZERO

#define OUTPUT_IMAGES 8
#define OUTPUT_RESULT 4
#define OUTPUT_SHADER_IMAGES 2
#define SHOW_VISUALISATION 1

//OpenCV 320-440 compatibility
// Video
#ifndef CV_CAP_IMAGES
  #define CV_CAP_IMAGES cv::CAP_IMAGES
#endif
#ifndef CV_CAP_PROP_FRAME_WIDTH
  #define CV_CAP_PROP_FRAME_WIDTH cv::CAP_PROP_FRAME_WIDTH
#endif
#ifndef CV_INTER_AREA
  #define CV_INTER_AREA cv::INTER_AREA
#endif
// ProcessedImage
#ifndef CV_FILLED
  #define CV_FILLED cv::FILLED
#endif
#ifndef CV_COVAR_NORMAL
  #define CV_COVAR_NORMAL cv::COVAR_NORMAL
#endif
#ifndef CV_COVAR_ROWS
  #define CV_COVAR_ROWS cv::COVAR_ROWS
#endif
#ifndef CV_COVAR_SCALE
  #define CV_COVAR_SCALE cv::COVAR_SCALE
#endif
#ifndef CV_RETR_EXTERNAL
  #define CV_RETR_EXTERNAL cv::RETR_EXTERNAL
#endif
#ifndef CV_CHAIN_APPROX_SIMPLE
  #define CV_CHAIN_APPROX_SIMPLE cv::CHAIN_APPROX_SIMPLE
#endif
//RegionsMatcher
#ifndef CV_COMP_CORREL
  #define CV_COMP_CORREL cv::HISTCMP_CORREL
#endif

#ifndef MAX_PATH
  #define MAX_PATH 260
#endif
