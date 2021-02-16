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
#version 330 core

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

float window_distance(vec2 coords1, vec2 coords2) {
  float result = 10.0;
  for (int i = kernel_lower; i < kernel_upper; i++)
    for (int j = kernel_lower; j < kernel_upper; j++)
      result = min(result, distance(
        texture2D(texColor1, coords1 + vec2(i,j) * pixel).xyz,
        texture2D(texColor2, coords2).xyz));
  return result;
}

void main() {
  vec3 projCoords1 = FragPosLightSpace1.xyz / FragPosLightSpace1.w;
  projCoords1 = projCoords1 * 0.5 + 0.5;
  float closestDepth1 = texture(shadowMap1, projCoords1.xy).r;
  vec3 projCoords2 = FragPosLightSpace2.xyz / FragPosLightSpace2.w;
  projCoords2 = projCoords2 * 0.5 + 0.5;
  float closestDepth2 = texture(shadowMap2, projCoords2.xy).r;

  float dist = (projCoords1.z > closestDepth1 ||
                projCoords2.z > closestDepth2) ? 0.0 :
                window_distance(projCoords1.xy, projCoords2.xy);
                //distance(texture2D(texColor1, projCoords1.xy).xyz,
                //         texture2D(texColor2, projCoords2.xy).xyz);

  color = vec4(vec3(dist), 1.0);
}
