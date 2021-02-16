/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "utils/otsu_miki.h"

// Modified from opencv/module/imgproc/src/thresh.cpp
//static double getThreshVal_Otsu_8u(const Mat& _src)

double otsu_8u_with_mask(const cv::Mat1b src, const cv::Mat1b& mask) {
    const int N = 256;
    int M = 0;
    int i, j, h[N] = { 0 };
    for (i = 0; i < src.rows; i++) {
        const uchar* psrc = src.ptr(i);
        const uchar* pmask = mask.ptr(i);
        for (j = 0; j < src.cols; j++)
            if (pmask[j]) {
                h[psrc[j]]++;
                ++M;
            }
    }

    double mu = 0, scale = 1. / (M);
    for (i = 0; i < N; i++)
        mu += i*(double)h[i];

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for (i = 0; i < N; i++) {
        double p_i, q2, mu2, sigma;
        p_i = h[i] * scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;
        if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON)
            continue;
        mu1 = (mu1 + i*p_i) / q1;
        mu2 = (mu - q1*mu1) / q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if (sigma > max_sigma) {
            max_sigma = sigma;
            max_val = i;
        }
    }
    return max_val;
}

double threshold_with_mask(cv::Mat1b& src, cv::Mat1b& dst, double thresh, double maxval, int type, const cv::Mat1b& mask) {
    if (mask.empty() || (mask.rows == src.rows && mask.cols == src.cols && countNonZero(mask) == src.rows * src.cols)) {
        // If empty mask, or all-white mask, use cv::threshold
        thresh = cv::threshold(src, dst, thresh, maxval, type);
    }
    else {
        // Use mask
        bool use_otsu = (type & cv::THRESH_OTSU) != 0;
        if (use_otsu) {
            // If OTSU, get thresh value on mask only
            thresh = otsu_8u_with_mask(src, mask);
            // Remove THRESH_OTSU from type
            type &= cv::THRESH_MASK;
        }
        // Apply cv::threshold on all image
        thresh = cv::threshold(src, dst, thresh, maxval, type);
        // Copy original image on inverted mask
        src.copyTo(dst, ~mask);
    }
    return thresh;
}

double triangle_8u_with_mask( const cv::Mat& _src ) {
    cv::Size size = _src.size();
    int step = (int) _src.step;
    if( _src.isContinuous() ) {
        size.width *= size.height;
        size.height = 1;
        step = size.width;
    }

    const int N = 256;
    int i, j, h[N] = {0};
    for( i = 0; i < size.height; i++ ) {
        const uchar* src = _src.ptr() + step*i;
        j = 0;
        #if CV_ENABLE_UNROLLED
        for( ; j <= size.width - 4; j += 4 ) {
            int v0 = src[j], v1 = src[j+1];
            h[v0]++; h[v1]++;
            v0 = src[j+2]; v1 = src[j+3];
            h[v0]++; h[v1]++;
        }
        #endif
        for( ; j < size.width; j++ )
            h[src[j]]++;
    }

    int left_bound = 0, right_bound = 0, max_ind = 0, max = 0;
    int temp;
    bool isflipped = false;

    //for( i = 0; i < N; i++ ) {
    for( i = 1; i < N; i++ ) {
        if( h[i] > 0 ) {
            left_bound = i;
            break;
        }
    }
    //if( left_bound > 0 )
    if( left_bound > 1 )
        left_bound--;

    //for( i = N-1; i > 0; i-- ) {
    for( i = N-1; i > 1; i-- ) {
        if( h[i] > 0 ) {
            right_bound = i;
            break;
        }
    }
    if( right_bound < N-1 )
        right_bound++;

    //for( i = 0; i < N; i++ ) {
    for( i = 1; i < N; i++ ) {
        if( h[i] > max) {
            max = h[i];
            max_ind = i;
        }
    }

    if( max_ind-left_bound < right_bound-max_ind) {
        isflipped = true;
        //i = 0, j = N-1;
        i = 1, j = N-1;
        while( i < j ) {
            temp = h[i]; h[i] = h[j]; h[j] = temp;
            i++; j--;
        }
        //left_bound = N-1-right_bound;
        left_bound = N-1-right_bound+1;
        max_ind = N-1-max_ind;
    }

    double thresh = left_bound;
    double a, b, dist = 0, tempdist;

    /*
     * We do not need to compute precise distance here. Distance is maximized, so some constants can
     * be omitted. This speeds up a computation a bit.
     */
    a = max; b = left_bound-max_ind;
    for( i = left_bound+1; i <= max_ind; i++ ) {
        tempdist = a*i + b*h[i];
        if( tempdist > dist) {
            dist = tempdist;
            thresh = i;
        }
    }
    thresh--;
    thresh += (N-1-thresh)/5; // offset for filtering more false positives

    if( isflipped )
        thresh = N-1-thresh;

    return thresh;
}

/*double getThreshVal_Otsu_8u( const cv::Mat& _src )
{
    cv::Size size = _src.size();
    int step = (int) _src.step;
    if( _src.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
        step = size.width;
    }

#ifdef HAVE_IPP
    unsigned char thresh;
    CV_IPP_RUN(IPP_VERSION_X100 >= 810, ipp_getThreshVal_Otsu_8u(_src.ptr(), step, size, thresh), thresh);
#endif

    const int N = 256;
    int i, j, h[N] = {0};
    for( i = 0; i < size.height; i++ )
    {
        const uchar* src = _src.ptr() + step*i;
        j = 0;
        #if CV_ENABLE_UNROLLED
        for( ; j <= size.width - 4; j += 4 )
        {
            int v0 = src[j], v1 = src[j+1];
            h[v0]++; h[v1]++;
            v0 = src[j+2]; v1 = src[j+3];
            h[v0]++; h[v1]++;
        }
        #endif
        for( ; j < size.width; j++ )
            h[src[j]]++;
    }*/
double getThreshVal_Otsu_8u(const cv::Mat1b _src) {
    const int N = 256;
    int i, j, h[N] = { 0 };

    /*for( i = 0; i < _src.rows; i++ )
    {
        const uchar* src = _src.ptr() + i;
        j = 0;
        #if CV_ENABLE_UNROLLED
        for( ; j <= _src.cols - 4; j += 4 )
        {
            int v0 = src[j], v1 = src[j+1];
            h[v0]++; h[v1]++;
            v0 = src[j+2]; v1 = src[j+3];
            h[v0]++; h[v1]++;
        }
        #endif
        for( ; j < _src.cols; j++ )
            h[src[j]]++;
    }*/

    for (i = 0; i < _src.rows; i++) {
        const uchar* psrc = _src.ptr(i);
        for (j = 0; j < _src.cols; j++)
            h[psrc[j]]++;
    }

    double mu = 0, scale = 1./(_src.cols*_src.rows);
    for( i = 0; i < N; i++ )
        mu += i*(double)h[i];

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for( i = 0; i < N; i++ )
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i]*scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if( std::min(q1,q2) < FLT_EPSILON || std::max(q1,q2) > 1. - FLT_EPSILON )
            continue;

        mu1 = (mu1 + i*p_i)/q1;
        mu2 = (mu - q1*mu1)/q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if( sigma > max_sigma )
        {
            max_sigma = sigma;
            max_val = i;
        }
    }

    return max_val;
}
