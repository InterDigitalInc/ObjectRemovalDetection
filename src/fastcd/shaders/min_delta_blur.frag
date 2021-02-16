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
 /*
 3x3 Median
 Morgan McGuire and Kyle Whitson
 http://graphics.cs.williams.edu


 Copyright (c) Morgan McGuire and Williams College, 2006
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are
 met:

 Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.

 Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#version 330 core

uniform sampler2D texOutput;
uniform bool initial;

uniform sampler2D texColor1;
uniform sampler2D shadowMap1;
in vec4 FragPosLightSpace1;

uniform sampler2D texColor2;
uniform sampler2D shadowMap2;
in vec4 FragPosLightSpace2;

uniform vec2 pixel;
uniform int kernel_lower;
uniform int kernel_upper;

out vec4 color;

#define s2(a, b)		 temp = a; a = min(a, b); b = max(temp, b);
#define mn3(a, b, c) s2(a, b); s2(a, c);
#define mx3(a, b, c) s2(b, c); s2(a, c);

#define mnmx3(a, b, c)			    mx3(a, b, c); s2(a, b);                                   // 3 exchanges
#define mnmx4(a, b, c, d)		    s2(a, b); s2(c, d); s2(a, c); s2(b, d);                   // 4 exchanges
#define mnmx5(a, b, c, d, e)	  s2(a, b); s2(c, d); mn3(a, c, e); mx3(b, d, e);           // 6 exchanges
#define mnmx6(a, b, c, d, e, f) s2(a, d); s2(b, e); s2(c, f); mn3(a, b, c); mx3(d, e, f); // 7 exchanges

float median_blur(vec2 coords) {
  float v[9];

  for(int i = -1; i <= 1; ++i)
    for(int j = -1; j <= 1; ++j) {
      vec2 offset = vec2(float(i), float(j));
      v[(i + 1) * 3 + (j + 1)] = texture2D(texOutput, coords + offset * pixel).r;
    }

  float temp;
  mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);
  mnmx5(v[1], v[2], v[3], v[4], v[6]);
  mnmx4(v[2], v[3], v[4], v[7]);
  mnmx3(v[3], v[4], v[8]);
  return v[4];
}

float window_distance(vec2 coords1, vec2 coords2) {
  float result = 10.0;
  for (int i = kernel_lower; i < kernel_upper; ++i)
    for (int j = kernel_lower; j < kernel_upper; ++j) {
      vec2 offset = vec2(float(i), float(j));
      result = min(result, distance(
        texture2D(texColor1, coords1 + offset * pixel).xyz,
        texture2D(texColor2, coords2).xyz));
    }
  return result;
}

void main() {
  vec3 projCoords1 = FragPosLightSpace1.xyz / (2.0 * FragPosLightSpace1.w);
  projCoords1 = projCoords1 + 0.5;
  float closestDepth1 = texture(shadowMap1, projCoords1.xy).r;
  vec3 projCoords2 = FragPosLightSpace2.xyz / (2.0 * FragPosLightSpace2.w);
  projCoords2 = projCoords2 + 0.5;
  float closestDepth2 = texture(shadowMap2, projCoords2.xy).r;

  float dist = (projCoords1.z > closestDepth1 ||
                projCoords2.z > closestDepth2) ? 0.0 :
                window_distance(projCoords1.xy, projCoords2.xy);

  dist = initial ? dist : min(dist, median_blur(projCoords2.xy));
  //dist = initial ? dist : min(dist, texture2D(texOutput, projCoords2.xy).r);
  color = vec4(vec3(dist), 1.0);
}
