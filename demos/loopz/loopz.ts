/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tslint:disable-next-line:max-line-length
import { Array1D, Array2D, Array3D, CheckpointLoader, NDArray, NDArrayMathGPU, Scalar } from '../deeplearn';
import { LSTMCell } from '../../src/math/math'
import * as demo_util from '../util';

const DECODER_CELL_FORMAT = "decoder/multi_rnn_cell/cell_%d/lstm_cell/"

let console: any = window.console;

const DEBUG = true;
if (!DEBUG) {
  console = {};
  console.log = () => { };
}
const forgetBias = Scalar.new(1.0);
let math: NDArrayMathGPU;

class LayerVars {
  kernel: Array2D;
  bias: Array1D;
  constructor(kernel: Array2D, bias: Array1D) {
    this.kernel = kernel;
    this.bias = bias;
  }
}

function dense(vars: LayerVars, inputs: Array2D) {
  const weightedResult = math.matMul(inputs, vars.kernel);
  return math.add(weightedResult, vars.bias) as Array2D;
}

class Encoder {
  lstmFwVars: LayerVars;
  lstmBwVars: LayerVars;
  muVars: LayerVars;
  presigVars: LayerVars;
  zDims: number;

  constructor(lstmFwVars: LayerVars, lstmBwVars: LayerVars, muVars: LayerVars, presigVars: LayerVars) {
    this.lstmFwVars = lstmFwVars;
    this.lstmBwVars = lstmBwVars;
    this.muVars = muVars;
    this.presigVars = presigVars;
    this.zDims = this.muVars.bias.shape[0];
  }

  private runLstm(inputs: Array3D, lstmVars: LayerVars, reverse: boolean) {
    const batchSize = inputs.shape[0];
    const length = inputs.shape[1];
    const outputSize = inputs.shape[2];
    let state: Array2D[] = [
      math.track(Array2D.zeros([batchSize, lstmVars.bias.shape[0] / 4])),
      math.track(Array2D.zeros([batchSize, lstmVars.bias.shape[0] / 4]))
    ]
    let lstm = math.basicLSTMCell.bind(math, forgetBias, lstmVars.kernel, lstmVars.bias);
    for (let i = 0; i < length; i++) {
      let index = reverse ? length - 1 - i : i;
      state = lstm(
        math.slice3D(inputs, [0, index, 0], [batchSize, 1, outputSize]).as2D(batchSize, outputSize),
        state[0], state[1]);
    }
    return state;
  }

  encode(sequence: Array3D): Array2D {
    const fwState = this.runLstm(sequence, this.lstmFwVars, false);
    const bwState = this.runLstm(sequence, this.lstmBwVars, true);
    const finalState = math.concat2D(fwState[1], bwState[1], 1)
    const mu = dense(this.muVars, finalState);
    return mu;
  }
}

class Decoder {
  lstmCellVars: LayerVars[];
  zToInitStateVars: LayerVars;
  outputProjectVars: LayerVars;
  zDims: number;
  outputDims: number;

  constructor(lstmCellVars: LayerVars[], zToInitStateVars: LayerVars, outputProjectVars: LayerVars) {
    this.lstmCellVars = lstmCellVars;
    this.zToInitStateVars = zToInitStateVars;
    this.outputProjectVars = outputProjectVars;
    this.zDims = this.zToInitStateVars.kernel.shape[0];
    this.outputDims = outputProjectVars.bias.shape[0];
  }

  decode(z: Array2D, length: number) {
    const batchSize = z.shape[0];
    const outputSize = this.outputProjectVars.bias.shape[0];

    // Initialize LSTMCells.
    let lstmCells: LSTMCell[] = []
    let c: Array2D[] = [];
    let h: Array2D[] = [];
    const initialStates = math.tanh(dense(this.zToInitStateVars, z));
    let stateOffset = 0;
    for (let i = 0; i < this.lstmCellVars.length; ++i) {
      const lv = this.lstmCellVars[i];
      const stateWidth = lv.bias.shape[0] / 4;
      lstmCells.push(math.basicLSTMCell.bind(math, forgetBias, lv.kernel, lv.bias))
      c.push(math.slice2D(initialStates, [0, stateOffset], [batchSize, stateWidth]));
      stateOffset += stateWidth;
      h.push(math.slice2D(initialStates, [0, stateOffset], [batchSize, stateWidth]));
      stateOffset += stateWidth;
    }

    // Generate samples.
    let samples: Array2D;
    let nextInput = math.track(Array2D.zeros([batchSize, outputSize]));
    for (let i = 0; i < length; ++i) {
      let output = math.multiRNNCell(lstmCells, math.concat2D(nextInput, z, 1), c, h);
      c = output[0];
      h = output[1];
      const logits = dense(this.outputProjectVars, h[h.length - 1]);

      let timeSamples = math.argMax(logits, 1).as1D();
      samples = i ? math.concat2D(samples, timeSamples.as2D(-1, 1), 1) : timeSamples.as2D(-1, 1);
      nextInput = math.oneHot(timeSamples, outputSize);
    }
    return samples;
  }
}

export const LoopZModel = {
  drums: './drums',
  melodies: './melodies',
  bigMelodies: './big_melodies'
}

export const isDeviceSupported = demo_util.isWebGLSupported() && !demo_util.isSafari();

class LoopZ {

  encoder: Encoder;
  decoder: Decoder;

  constructor(checkpointURL: string, callback: Function) {
    if (!isDeviceSupported) {
      callback(new Error("device is not supported"), this);
      return;
    }

    math = new NDArrayMathGPU();
    // math.enableDebugMode()

    this.initialize(checkpointURL).then((encoder_decoder: [Encoder, Decoder]) => {
      this.encoder = encoder_decoder[0];
      this.decoder = encoder_decoder[1];
      callback(null, this);
    });
  }

  initialize(checkpointURL: string) {
    const reader = new CheckpointLoader(checkpointURL);
    return reader.getAllVariables().then(
      (vars: { [varName: string]: NDArray }) => {
        const encLstmFw = new LayerVars(
          vars['encoder/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/kernel'] as Array2D,
          vars['encoder/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/bias'] as Array1D);
        const encLstmBw = new LayerVars(
          vars['encoder/cell_0/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/kernel'] as Array2D,
          vars['encoder/cell_0/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/bias'] as Array1D);
        const encMu = new LayerVars(
          vars['encoder/mu/kernel'] as Array2D,
          vars['encoder/mu/bias'] as Array1D);
        const encPresig = new LayerVars(
          vars['encoder/sigma/kernel'] as Array2D,
          vars['encoder/sigma/bias'] as Array1D);

        let decLstmLayers: LayerVars[] = [];
        let l = 0;
        while (true) {
          const cell_prefix = DECODER_CELL_FORMAT.replace('%d', l.toString());
          if (!(cell_prefix + 'kernel' in vars)) {
            break;
          }
          decLstmLayers.push(new LayerVars(
            vars[cell_prefix + 'kernel'] as Array2D,
            vars[cell_prefix + 'bias'] as Array1D));
          ++l;
        }

        const decZtoInitState = new LayerVars(
          vars['decoder/z_to_initial_state/kernel'] as Array2D,
          vars['decoder/z_to_initial_state/bias'] as Array1D);
        const decOutputProjection = new LayerVars(
          vars['decoder/output_projection/kernel'] as Array2D,
          vars['decoder/output_projection/bias'] as Array1D);
        return [
          new Encoder(encLstmFw, encLstmBw, encMu, encPresig),
          new Decoder(decLstmLayers, decZtoInitState, decOutputProjection)];
      })
  }

  // optimized encoding by Nikhil and Daniel
  async interpolate(noteSequences: number[][], numSteps: number) {
    if (noteSequences.length != 2 && noteSequences.length != 4) {
      throw new Error('invalid number of note sequences. Requires length 2, or 4');
    }
    
    const z = math.scope((keep, track) => {
      // One-hot encode
      const startSeq = track(Array1D.new(noteSequences[0]));

      const oneHotStartSeq = math.oneHot(startSeq, this.decoder.outputDims);
      const oneHotStartSeq3D = oneHotStartSeq.as3D(
        1, oneHotStartSeq.shape[0], oneHotStartSeq.shape[1]);

      let batchedInput: Array3D = oneHotStartSeq3D;
      for (let i = 1; i < noteSequences.length; i++) {
        const endSeq = track(Array1D.new(noteSequences[1]));
        const oneHotEndSeq = math.oneHot(endSeq, this.decoder.outputDims);
        const oneHotEndSeq3D = oneHotEndSeq.as3D(
          1, oneHotEndSeq.shape[0], oneHotEndSeq.shape[1]);
        batchedInput = math.concat3D(batchedInput, oneHotEndSeq3D, 0);
      }
      // Compute z values.
      return this.encoder.encode(batchedInput);
    });

    // Interpolate.
    const range: number[] = [];
    for (let i = 0; i < numSteps; i++) {
      range.push(i / (numSteps - 1));
    }

    const interpolatedZs = await math.scope(async (keep, track) => {
      const rangeArray = track(Array1D.new(range).as2D(range.length, 1));

      const z0 = math.slice2D(z, [0, 0], [1, z.shape[1]]);
      const z1 = math.slice2D(z, [1, 0], [1, z.shape[1]]);

      const zDiff = math.subtract(z1, z0);
      const diffRange = math.multiply(zDiff, rangeArray) as Array2D; //brodcasting
      const interpolant1 = math.add(diffRange, z0) as Array2D;
      if (noteSequences.length == 2) {
        return interpolant1;
      } else if (noteSequences.length == 4) {
        const z2 = math.slice2D(z, [2, 0], [1, z.shape[1]]);
        const z3 = math.slice2D(z, [3, 0], [1, z.shape[1]]);

        const zDiff2 = math.subtract(z3, z2);
        const diffRange2 = math.multiply(zDiff2, rangeArray) as Array2D; //brodcasting
        const interpolant2 = math.add(diffRange2, z2) as Array2D;

        const finalDiff = math.subtract(interpolant2, interpolant1);
        const finalDiffRange = math.multiply(finalDiff, rangeArray);
        return math.add(finalDiffRange, interpolant1) as Array2D;

      }
    });

    return math.scope(() => {
      return this.decoder.decode(interpolatedZs, noteSequences[0].length);
    });
  }
}

//this will be handled by the prototype
new LoopZ('./drums', async (err: Error, loopZ: LoopZ) => {
  if (err != null) {
    throw err;
  }

  const teaPot = [71, 0, 73, 0, 75, 0, 76, 0, 78, 0, 1, 0, 83, 0, 0, 0, 80, 0, 0, 0, 83, 0, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0];
  const teaPots: [number[], number[], number[], number[]] = [teaPot, teaPot.slice(0).reverse(), teaPot, teaPot.slice(0).reverse()];
  // const teaPot = [71, 0, 73, 0, 75, 0, 76, 0, 78, 0, 1, 0, 83, 0, 0, 0, 80, 0, 0, 0, 83, 0, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0];
  // const teaPots: [number[], number[]] = [teaPot, teaPot.slice(0).reverse()];

  const drums = [[2, 1, 5, 1, 2, 1, 5, 1, 2, 1, 5, 1, 2, 1, 5, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 5, 1],
  [16, 1, 5, 5, 69, 1, 24, 1, 14, 1, 101, 1, 40, 65, 53, 1, 144, 1, 5, 5, 69, 1, 24, 1, 14, 1, 101, 1, 8, 1, 37, 1],
  [7, 1, 5, 1, 5, 1, 24, 1, 6, 1, 5, 1, 23, 1, 14, 1, 6, 1, 5, 1, 23, 1, 6, 1, 6, 1, 23, 1, 21, 1, 22, 1],
  [1, 1, 1, 1, 65, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 65, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]];

  var start = Date.now();

  let data = await loopZ.encodeAndDecodeGPU(teaPots, 90);
  console.log('optimized: ' + (Date.now() - start) / 1000.);

  for (let i = 0; i < data.shape[0]; i++) {
    const r = math.slice2D(data, [i, 0], [1, data.shape[1]]);
    await r.data()
  }
  console.log('gpu format data: ' + (Date.now() - start) / 1000.);
});

export { LayerVars, Encoder, Decoder, LoopZ }
