/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {initializeGPU} from '../../src/math/ndarray';
import {Conv2DDerInputProgram} from '../../src/math/webgl/conv_backprop_gpu';
import * as gpgpu_math from '../../src/math/webgl/gpgpu_math';
import {TextureManager} from '../../src/math/webgl/texture_manager';
// tslint:disable-next-line:max-line-length
import {Array3D, Array4D, conv_util, ENV, GPGPUContext} from '../deeplearn';

import {BenchmarkTest} from './benchmark';

export interface ConvTransposedBenchmarkParams {
  inDepth: number;
  outDepth: number;
  filterSize: number;
  stride: number;
}

export abstract class ConvTransposedBenchmark extends BenchmarkTest {
  constructor(protected params: ConvTransposedBenchmarkParams) {
    super(params);
  }
}

export class ConvTransposedGPUBenchmark extends ConvTransposedBenchmark {
  async run(size: number): Promise<number> {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const xShape: [number, number, number] = [size, size, origOutputDepth];
    const fieldSize = 11;
    const origStride = 1;
    const origPad = 1;

    const gpgpu = new GPGPUContext();
    const texManager = new TextureManager(gpgpu);
    initializeGPU(gpgpu, texManager);
    gpgpu.enableAutomaticDebugValidation(true);

    const convInfo = conv_util.computeConvInfo(
        xShape, fieldSize, fieldSize, origOutputDepth, origStride, origStride,
        origPad);
    const program = new Conv2DDerInputProgram(convInfo);
    const outputShape = program.outputShape as [number, number, number];
    const out = Array3D.zeros(outputShape);
    const x = Array3D.randUniform(xShape, -1, 1);
    const wShape = conv_util.computeWeightsShape4D(
        origInputDepth, origOutputDepth, fieldSize, fieldSize);
    const W = Array4D.randUniform(wShape, -1, 1);
    const inputs = [x, W];
    const binary = gpgpu_math.compileProgram(gpgpu, program, inputs, out);

    const benchmark = () => {
      gpgpu_math.runProgram(binary, inputs, out);
    };

    const cleanup = () => {
      out.dispose();
      x.dispose();
      W.dispose();
      texManager.dispose();
      gpgpu.deleteProgram(binary.webGLProgram);
      gpgpu.dispose();
    };

    // Warmup.
    await gpgpu.runQuery(benchmark);

    let totalTime: number;
    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) {
      totalTime = await gpgpu.runQuery(benchmark);
    } else {
      const start = performance.now();

      benchmark();
      out.dataSync();

      totalTime = performance.now() - start;
    }
    cleanup();
    return totalTime;
  }
}
