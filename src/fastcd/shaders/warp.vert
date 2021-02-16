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
layout (location = 0) in vec4 position_in;

uniform mat4 mvp;         // model-view-projection matrix. (applied p*v*m)

uniform mat4 mvp1;         // model-view-projection matrix for the texture 1
out vec4 FragPosLightSpace1;

void main() {
  gl_Position = mvp * position_in;
  FragPosLightSpace1 = mvp1 * position_in;
}
