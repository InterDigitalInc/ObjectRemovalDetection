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
#include <opencv2/opencv.hpp>
#include <iostream>

double otsu_8u_with_mask(const cv::Mat1b src, const cv::Mat1b& mask);

double threshold_with_mask(cv::Mat1b& src, cv::Mat1b& dst, double thresh, double maxval, int type, const cv::Mat1b& mask = cv::Mat1b());

//double getThreshVal_Otsu_8u( const cv::Mat& _src );
double getThreshVal_Otsu_8u(const cv::Mat1b src);

double triangle_8u_with_mask( const cv::Mat& _src );
