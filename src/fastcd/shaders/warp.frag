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

out vec4 color;

void main() {
  vec3 projCoords1 = FragPosLightSpace1.xyz / FragPosLightSpace1.w;
  projCoords1 = projCoords1 * 0.5 + 0.5;
  float closestDepth1 = texture(shadowMap1, projCoords1.xy).r;
  float visible1 = projCoords1.z > closestDepth1 ? 0.0 : 1.0;
  color = vec4(visible1 * texture2D(texColor1, projCoords1.xy).xyz, 1.0);
}
